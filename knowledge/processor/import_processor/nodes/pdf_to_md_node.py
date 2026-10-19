"""PDF 转 Markdown 节点：调用 MinerU 解析 PDF，轮询结果、下载 ZIP 并提取 full.md。"""
import logging
import shutil
import time
import zipfile
from pathlib import Path
import requests
from knowledge.processor.import_processor.base import BaseNode, T
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import ValidationError
from knowledge.processor.import_processor.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 上传 PDF 并轮询 MinerU 解析结果，返回结果 ZIP 地址
        self.log_step(step_name="Step1", message="上传PDF到MinerU并轮询解析结果")
        pdf_path = state.get("pdf_path")
        pdf_path_obj = Path(pdf_path)
        zip_url = self._upload_pdf_and_query_result(self.config,pdf_path_obj)

        # 下载 ZIP 并解压出 md 文件
        self.log_step(step_name="Step2", message="下载Zip并解压")
        file_dir = state.get("file_dir")
        file_dir_obj = Path(file_dir)
        md_path = self._extract_md(zip_url,file_dir_obj,pdf_path_obj)
        state['md_path'] = md_path
        return state

    def _upload_pdf_and_query_result(self,config:ImportConfig,pdf_path_obj:Path) -> str:
        # 校验 MinerU 配置
        mineru_api_token = config.mineru_api_token
        mineru_base_url = config.mineru_base_url
        if not mineru_api_token or not mineru_base_url:
            self.logger.error("mineru_api_token or mineru_base_url not set")
            raise ValidationError(node_name=self.name,message="mineru_api_token or mineru_base_url not set")

        # 申请文件上传链接
        url = mineru_base_url + "/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {mineru_api_token}"
        }

        data = {
            "files": [
                {"name": f"{pdf_path_obj.name}", "data_id": "abcd"}
            ],
            "model_version": "vlm"
        }
        try:
            response = requests.post(url, headers=header, json=data)
        except Exception as e:
            self.logger.error(f"【获取上传链接】接口调用失败，异常信息：{str(e)}")
            raise RuntimeError(f"【获取上传链接】接口调用失败，异常信息：{str(e)}")

        if response.status_code != 200:
            self.logger.error(f"【获取上传链接】接口调用失败，状态码：{response.status_code}，响应内容：{response}")
            raise RuntimeError(f"【获取上传链接】接口调用失败，状态码：{response.status_code}，响应内容：{response}")

        result = response.json()
        if result["code"] != 0:
            self.logger.error(f"【获取上传链接】获取链接失败，业务状态码：{result["code"]}，响应内容：{result}")
            raise RuntimeError(f"【获取上传链接】获取链接失败，业务状态码：{result["code"]}，响应内容：{result}")
        batch_id = result["data"]["batch_id"]
        upload_url = result["data"]["file_urls"][0]
        self.logger.info(f"【获取上传链接】成功获取到上传链接，batch_id:{batch_id}, upload_url:{upload_url}")

        # PUT 上传 PDF 到 MinerU
        try:
            with open(str(pdf_path_obj), "rb") as f:
                file_content = f.read()
                res_upload = requests.put(upload_url, data=file_content)
        except Exception as e:
            self.logger.error(f"【上传PDF】上传PDF失败，异常信息：{str(e)}")
            raise RuntimeError(f"【上传PDF】上传PDF失败，异常信息：{str(e)}")

        if res_upload.status_code != 200:
            self.logger.error(f"【上传PDF】上传PDF失败，状态码：{res_upload.status_code}，响应内容：{res_upload}")
            raise RuntimeError(f"【上传PDF】上传PDF失败，状态码：{res_upload.status_code}，响应内容：{res_upload}")

        self.logger.info(f"【上传PDF】成功上传PDF文件到MinerU，文件名：{pdf_path_obj.name}，batch_id:{batch_id}")

        # 轮询解析结果：超时 200s，间隔 3s，直到 done / failed
        max_time = 200
        interval_time = 3
        start_time = time.time()
        while True:
            end_time = time.time()
            if end_time - start_time > max_time:
                self.logger.warning(f"【轮询解析结果】轮询超时，轮询时间：{end_time - start_time:.2f}s，batch_id: {batch_id}")
                break
            pull_url = mineru_base_url + f"/extract-results/batch/{batch_id}"
            pull_res = requests.get(pull_url, headers=header)
            # 非 200 则休眠后重试
            if pull_res.status_code != 200:
                self.logger.warning(f"【轮询解析结果】失败, 状态码为:{pull_res.status_code}")
                time.sleep(interval_time)
                continue
            pull_result = pull_res.json()
            if pull_result['code'] != 0:
                self.logger.warning(f"【轮询解析结果】业务失败, 业务状态码为:{pull_result['code']}")
                time.sleep(interval_time)
                continue

            extract_result = pull_result['data']['extract_result']
            extract_state = extract_result[0]['state']
            if extract_state == "done":
                # 解析完成，返回结果包地址
                full_zip_url = extract_result[0]['full_zip_url']
                self.logger.info(f"【轮询解析结果】MinerU解析成功, batch_id:{batch_id}, full_zip_url:{full_zip_url}")
                return full_zip_url
            elif extract_state == "failed":
                self.logger.error(f"【轮询解析结果】MinerU解析失败, batch_id:{batch_id}")
                raise RuntimeError(f"【轮询解析结果】MinerU解析失败, batch_id:{batch_id}")
            else:
                # 其它中间状态，继续轮询
                time.sleep(interval_time)
                continue

    def _extract_md(self, zip_url:str,file_dir_obj:Path,pdf_file_obj:Path) -> str:
        # 下载结果 ZIP
        try:
            res = requests.get(zip_url, timeout=20)
        except Exception as e:
            self.logger.error(f"下载zip包失败,失败信息是{str(e)}")
            raise RuntimeError(f"下载zip包失败,失败信息是{str(e)}")

        if res.status_code != 200:
            self.logger.error(f"下载zip包失败，状态码：{res.status_code}，响应内容：{res}")
            raise RuntimeError(f"下载zip包失败，状态码：{res.status_code}，响应内容：{res}")
        self.logger.info(f"成功下载zip包，zip_url:{zip_url}，文件名：{pdf_file_obj.name}")

        # 保存为 file_dir/<pdf名>_result.zip
        zip_path = file_dir_obj / f"{pdf_file_obj.stem}_result.zip"
        try:
            with open(zip_path, "wb") as f:
                f.write(res.content)
        except Exception as e:
            self.logger.error(f"保存zip包失败,失败信息是{str(e)}")
            raise RuntimeError(f"保存zip包失败,失败信息是{str(e)}")

        self.logger.info(f"成功保存zip包，zip_path:{zip_path}")

        # 解压到 file_dir/<pdf名>/，已有内容先清空
        extract_path = file_dir_obj / f"{pdf_file_obj.stem}"
        try:
            if extract_path.exists():
                shutil.rmtree(extract_path)
        except Exception as e:
            self.logger.warning(f"清除解压目录失败,失败信息是{str(e)}, 目录路径是{extract_path}，但不影响后续解压")
        try:
            with zipfile.ZipFile(zip_path,"r") as zip_ref:
                zip_ref.extractall(extract_path)
        except Exception as e:
            self.logger.error(f"解压zip包失败,失败信息是{str(e)}")
            raise RuntimeError(f"解压zip包失败,失败信息是{str(e)}")

        self.logger.info(f"解压zip包成功,解压目录为{extract_path}")

        # 将 full.md 重命名为 <pdf名>.md
        md_file_path = extract_path / "full.md"
        new_md_file_path = extract_path / f"{pdf_file_obj.stem}.md"
        md_file_path.rename(new_md_file_path)

        return str(new_md_file_path)

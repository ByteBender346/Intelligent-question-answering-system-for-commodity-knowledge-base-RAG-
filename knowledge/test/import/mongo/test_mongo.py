"""MongoDB 增删改查离线测试脚本。"""

from pymongo import MongoClient


def test_create_collection():
    # 连接 MongoDB
    client = MongoClient("mongodb://admin:123456@192.168.150.120:27017")

    # 选择数据库与集合（不存在则自动创建）
    db = client["mydb"]
    collection = db["students"]

    print("连接成功！")
    return collection

def test_insert(collection):
    result = collection.insert_one({
    "name": "张三",
    "age": 20,
    "major": "计算机科学"
    })
    print(f"插入成功，ID: {result.inserted_id}")
    # 插入多条
    results = collection.insert_many([
        {"name": "李四", "age": 22, "major": "软件工程"},
        {"name": "王五", "age": 21, "major": "计算机科学"},
    ])
    print(f"插入 {len(results.inserted_ids)} 条记录")

def test_query(collection):
    # 按 age 倒序取前 2 条
    for doc in collection.find().sort("age", -1).limit(2):
        print(doc["name"], doc["age"])

def test_update(collection):
    # 批量更新专业为计算机科学的记录
    result = collection.update_many(
        {"major": "计算机科学"},
        {"$set": {"status": "在读"}}
    )
    print(f"修改 {result.modified_count} 条")

def test_delete(collection):
    # 删除年龄小于 21 的记录
    result = collection.delete_many({"age": {"$lt": 21}})
    print(f"删除 {result.deleted_count} 条")

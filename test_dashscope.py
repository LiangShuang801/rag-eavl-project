import os
import dashscope
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")

print("API Key是否读取成功：", bool(api_key))

dashscope.api_key = api_key

response = dashscope.Generation.call(
    model="qwen-turbo",
    messages=[
        {
            "role": "user",
            "content": "你好，请回复：百炼连接成功"
        }
    ],
    result_format="message"
)

print(response)
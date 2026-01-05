import requests
import base64
import os
import extract
import QW
SAVE_IMG_PATH = "test1111.jpg"
folder = r"D:\\qwen3_deploy\samTestImage"  # 存放图片文件夹的绝对路径

def send_data_to_server(x1, y1, x2, y2, x3, y3, x4, y4, img_path, server_url):
    # 1. 构造请求数据：
    # - data：存放数字常数等键值对数据（表单格式）
    # - files：存放图片文件（二进制流）
    request_data = {
        "x1": x1,  # 数字常数，自动转为字符串传输
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "x3": x3,
        "y3": y3,
        "x4": x4,
        "y4": y4,
    }
    request_files = {
        "img_file": open(img_path, "rb")  # 二进制读取图片
    }

    # 2. 发送POST请求（核心：multipart/form-data格式）
    try:
        res = requests.post(server_url, data=request_data, files=request_files)
        if res.status_code == 200:
            print("✅ 发送成功！服务端响应：")
            # 2.2 提取文件数据 → 解码 → 保存到磁盘
            img_base64 = res.json()["file_base64"]
            img_bytes = base64.b64decode(img_base64)  # Base64解码为二进制
            print(img_bytes)
            try:
                with open(SAVE_IMG_PATH, "wb") as f:  # 二进制写入磁盘
                    f.write(img_bytes)
            except Exception as e:
                print(str(e))

            print(f"\n✅ 操作完成！")
            print(f"📄 图片已保存至：{os.path.abspath(SAVE_IMG_PATH)}")

        else:
            print(f"❌ 发送失败，状态码：{res.status_code}，响应：{res.text}")
    except Exception as e:
        print(f"❌ 请求异常：{str(e)}")
    finally:
        request_files["img_file"].close()  # 关闭文件句柄


def work(
        x1=100,  # 待发送的数字常数
        y1=300,
        x2=750,
        y2=1000,
        x3=50,
        y3=50,
        x4=900,
        y4=1050,
        IMG_PATH="apple.jpg",  # 待发送的图片路径
        SERVER_API_URL="http://192.168.37.193:8080/receive_data"  # 服务端接口地址
):
    send_data_to_server(x1, y1, x2, y2, x3, y3, x4, y4, IMG_PATH, SERVER_API_URL)


if __name__ == "__main__":

    for filename in os.listdir(folder):  # 遍历文件夹中的文件
        full_path = os.path.join(folder, filename)
        print(full_path)

        # Todo 调用qwen 获取数字参数
        SAVE_IMG_PATH=filename
        num_list= extract.extract(QW.QWen3(full_path))
        x1=num_list[0]
        y1=num_list[1]
        x2=num_list[2]
        y2=num_list[3]
        x3=num_list[4]
        y3=num_list[5]
        x4=num_list[6]
        y4=num_list[7]
        work(
            x1=x1,  # 待发送的数字常数
            y1=y1,
            x2=x2,
            y2=y2,
            x3=x3,
            y3=y3,
            x4=x4,
            y4=y4,
            IMG_PATH=full_path,  # 待发送的图片路径
            SERVER_API_URL="http://192.168.37.193:8080/receive_data"  # 服务端接口地址
        )
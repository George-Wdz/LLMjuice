import os
import time
import requests
import json
import zipfile
import io
import urllib3
from pathlib import Path
from dotenv import load_dotenv

# ================= 配置区域 =================
# 1. 加载环境变量
load_dotenv()
token = os.getenv("MinerU_KEY")
if not token:
    print("❌ 错误: 未在 .env 文件中配置 MinerU_KEY")
    exit(1)

# 2. 路径配置
base_dir = Path(__file__).parent
pdf_dir = base_dir / "data" / "pdf"
md_dir = base_dir / "data" / "markdown"
md_dir.mkdir(parents=True, exist_ok=True)

# 3. 网络配置 (WSL/代理兼容)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# ================= 核心功能函数 =================

def get_files_to_process():
    """扫描目录，跳过已处理的文件"""
    if not pdf_dir.exists():
        print(f"❌ PDF 目录不存在: {pdf_dir}")
        return []

    pending_files = []
    print(f"📂 正在扫描目录: {pdf_dir} ...")
    
    for pdf_file in pdf_dir.glob("*.pdf"):
        # 检查是否已存在同名结果文件夹或md文件
        # 现在的下载逻辑是建立子文件夹，所以我们检查子文件夹是否存在
        result_folder = md_dir / pdf_file.stem
        
        if result_folder.exists() and any(result_folder.iterdir()):
            print(f"⏩ 跳过 (已解析): {pdf_file.name}")
            continue
        
        pending_files.append(pdf_file)
        
    return pending_files

def upload_files(files_to_upload):
    """批量申请链接并上传文件"""
    print(f"\n🚀 开始处理 {len(files_to_upload)} 个新文件...")

    # 1. 构造请求参数
    request_file_list = []
    for f in files_to_upload:
        request_file_list.append({
            "name": f.name,
            "data_id": f.stem 
        })

    # 2. 申请上传链接
    url_get_links = "https://mineru.net/api/v4/file-urls/batch"
    payload = {
        "files": request_file_list,
        "model_version": "vlm"
    }

    try:
        print("📡 正在申请上传链接...")
        res = requests.post(url_get_links, headers=headers, json=payload, verify=False)
        
        if res.status_code != 200:
            print(f"❌ 请求失败: {res.text}")
            return None

        res_json = res.json()
        if res_json["code"] != 0:
            print(f"❌ API 错误: {res_json['msg']}")
            return None

        batch_id = res_json["data"]["batch_id"]
        file_urls = res_json["data"]["file_urls"]
        print(f"✅ 批次创建成功! Batch ID: {batch_id}")

        # 3. 上传文件
        success_count = 0
        for i, upload_url in enumerate(file_urls):
            local_file_path = files_to_upload[i]
            print(f"⬆️  正在上传 ({i+1}/{len(file_urls)}): {local_file_path.name}")
            
            with open(local_file_path, "rb") as f:
                # PUT 上传，无需特定 Header
                upload_res = requests.put(upload_url, data=f, verify=False)
                if upload_res.status_code == 200:
                    success_count += 1
                else:
                    print(f"   ❌ 上传失败 (Status: {upload_res.status_code})")
        
        if success_count == 0:
            print("❌ 所有文件上传失败，终止流程。")
            return None
            
        return batch_id

    except Exception as e:
        print(f"❌ 上传阶段发生异常: {e}")
        return None

def monitor_and_download(batch_id):
    """轮询状态并下载结果"""
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    print(f"\n🔍 开始轮询解析结果 (Batch: {batch_id})")
    print("⏳ 系统正在解析中，请稍候...")

    # 记录已下载的文件，避免重复下载
    downloaded_files = set()

    while True:
        try:
            time.sleep(5) # 每5秒查一次
            res = requests.get(url, headers=headers, verify=False)
            
            if res.status_code != 200:
                print(f"⚠️ 查询请求失败: {res.status_code}，重试中...")
                continue
            
            data = res.json()
            if data["code"] != 0:
                print(f"❌ 查询返回错误: {data['msg']}")
                break
            
            extract_results = data["data"]["extract_result"]
            
            # 统计当前批次的状态
            all_finished = True
            running_cnt = 0
            
            # 简单进度条显示
            status_summary = []

            for item in extract_results:
                state = item["state"]
                fname = item["file_name"]
                
                if state == "running" or state == "pending" or state == "waiting-file":
                    all_finished = False
                    running_cnt += 1
                
                # 如果状态是 done 且 还没下载过，立即下载
                if state == "done" and fname not in downloaded_files:
                    print(f"\n✅ 检测到完成: {fname}，正在下载...")
                    if download_single_file(item):
                        downloaded_files.add(fname)
                
                # 如果失败了
                if state == "failed" and fname not in downloaded_files:
                    print(f"\n❌ 解析失败: {fname} (原因: {item.get('err_msg')})")
                    downloaded_files.add(fname) # 标记为已处理，不再报错

            # 打印简略进度
            print(f"\r⏳ 剩余任务: {running_cnt} 个正在处理...", end="")

            if all_finished:
                print("\n\n🎉 当前批次所有任务处理完毕！")
                break

        except KeyboardInterrupt:
            print("\n🛑 用户手动停止轮询。")
            break
        except Exception as e:
            print(f"\n❌ 轮询异常: {e}")
            break

def download_single_file(item_data):
    """下载单个文件的ZIP并解压"""
    try:
        zip_url = item_data.get("full_zip_url")
        file_name = item_data.get("file_name")
        
        if not zip_url:
            return False

        # 创建输出目录：data/markdown/文件名/
        folder_name = Path(file_name).stem
        output_folder = md_dir / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)

        # 下载
        zip_res = requests.get(zip_url, verify=False)
        
        # 解压
        with zipfile.ZipFile(io.BytesIO(zip_res.content)) as z:
            z.extractall(output_folder)
            
        print(f"   💾 已保存到: {output_folder}")
        return True
    except Exception as e:
        print(f"   ❌ 下载解压出错: {e}")
        return False

# ================= 主程序入口 =================

if __name__ == "__main__":
    # 1. 获取需要处理的文件
    files = get_files_to_process()
    
    if not files:
        print("😴 没有发现新文件，程序退出。")
    else:
        # 2. 上传文件并获取 Batch ID
        batch_id = upload_files(files)
        
        if batch_id:
            # 3. 如果上传成功，立即开始轮询下载
            # 等待几秒让服务器反应一下
            time.sleep(2)
            monitor_and_download(batch_id)
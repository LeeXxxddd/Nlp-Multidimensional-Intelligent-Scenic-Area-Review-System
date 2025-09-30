import requests
import json

BASE_URL = "http://connect.yza1.seetacloud.com:57348"

# --- 测试健康检查接口 ---
def test_health():
    url = f"{BASE_URL}/health"
    print(f"Testing GET: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        print("Response from /health:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except requests.exceptions.RequestException as e:
        print(f"Error testing /health: {e}")

# --- 测试评论评估接口 ---
def test_evaluate_comment(comment_text):
    url = f"{BASE_URL}/evaluate_comment"
    headers = {"Content-Type": "application/json"}
    payload = {"comment": comment_text}
    print(f"\nTesting POST: {url}")
    print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print("Response from /evaluate_comment:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except requests.exceptions.RequestException as e:
        print(f"Error testing /evaluate_comment: {e}")

if __name__ == "__main__":
    test_health()

    # 测试一个广告评论
    test_evaluate_comment("专业代写各类论文，价格合理，保质保量，详情加扣扣：123456789。")

    # 测试一个非广告评论
    test_evaluate_comment("风景很美，值得一去。")

    # 测试一个修改后的评论
    test_evaluate_comment("专业代写论文，合理价，包通过，Q号：123456789。")
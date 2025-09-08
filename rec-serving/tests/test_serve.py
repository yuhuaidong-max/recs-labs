import requests

# 我们的服务在 Codespace 中运行时，可以通过 localhost 访问
BASE_URL = "http://localhost:8000"

def test_health_check():
    """测试 /health 接口是否正常工作"""
    response = requests.get(f"{BASE_URL}/health")
    # 断言1：状态码必须是 200 (OK)
    assert response.status_code == 200
    # 断言2：返回的 JSON 内容必须符合预期
    assert response.json() == {"status": "ok"}

def test_recommend_api():
    """测试 /recommend 接口是否能对正常的用户ID返回预期结构"""
    user_id = 1
    response = requests.get(f"{BASE_URL}/recommend?user_id={user_id}")
    
    # 断言1：状态码必须是 200 (OK)
    assert response.status_code == 200
    
    # 断言2：返回的是有效的 JSON
    data = response.json()
    
    # 断言3：JSON 的结构符合预期
    assert "user_id" in data
    assert "recommendations" in data
    assert data["user_id"] == user_id
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) > 0
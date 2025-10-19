import streamlit as st
import requests
import json

# 页面配置
st.set_page_config(
    page_title="Ollama 对话助手",
    page_icon="🤖",
    layout="wide"
)

# 标题
st.title("Ollama × Streamlit 对话界面")
st.caption("与本地Ollama模型进行交互")

# 侧边栏配置
with st.sidebar:
    st.header("配置")
    # Ollama服务地址（默认端口11434）
    ollama_base_url = st.text_input(
        "Ollama API地址",
        value="http://localhost:11434",
        help="默认本地地址为 http://localhost:11434"
    )
    # 选择模型（需提前通过ollama pull下载）
    model_name = st.text_input(
        "模型名称",
        value="qwen3:0.6b",  # 可替换为其他模型如gemma、qwen等
        help="需填写Ollama中已存在的模型名称"
    )
    # 对话参数
    temperature = st.slider("温度（创造性）", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("最大回复长度", 100, 2000, 500, 100)

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用Ollama API生成回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 构建请求参数
        url = f"{ollama_base_url}/api/chat"
        payload = {
            "model": model_name,
            "messages": st.session_state.messages,  # 传入完整对话历史
            "stream": True,  # 流式返回（逐字显示）
            "options": {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }

        try:
            # 发送POST请求（流式响应）
            with requests.post(url, json=payload, stream=True) as r:
                r.raise_for_status()  # 检查请求是否成功
                for line in r.iter_lines():
                    if line:
                        # 解析流式返回的JSON
                        data = json.loads(line.decode("utf-8"))
                        # 提取当前片段内容（处理结束标记）
                        if "message" in data and "content" in data["message"]:
                            chunk = data["message"]["content"]
                            full_response += chunk
                            # 实时更新显示
                            message_placeholder.markdown(full_response + "▌")
                # 移除光标
                message_placeholder.markdown(full_response)
                
                # 添加助手回复到历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

        except requests.exceptions.ConnectionError:
            st.error("无法连接到Ollama服务，请检查：\n1. Ollama是否已启动\n2. API地址是否正确（默认端口11434）")
        except Exception as e:
            st.error(f"发生错误：{str(e)}")

# conda activate NLPAPI
# python -m streamlit run D:\Codes\NLPAPI\API.py

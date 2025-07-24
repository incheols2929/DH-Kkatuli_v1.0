from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from llama_cpp import Llama
import json
import asyncio
import logging
import os
import psutil

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 모델을 전역 변수로 선언하되 초기화는 지연
llm = None


def optimize_system():
    """시스템 최적화 설정"""
    # CPU 친화도 설정 (모든 코어 사용)
    try:
        process = psutil.Process()
        process.cpu_affinity(list(range(psutil.cpu_count())))
        # 우선순위 높음으로 설정
        process.nice(psutil.HIGH_PRIORITY_CLASS)
    except:
        pass


def initialize_model():
    """모델을 안전하게 초기화"""
    global llm
    if llm is None:
        try:
            logger.info("모델 로딩 시작...")
            llm = Llama(
                #model_path="models/EEVE-Korean-Instruct-10.8B-v1.0-Q8_0.gguf",
                # model_path="models/Gugugo-koen-7B-V1.1.Q8_0.gguf",
                # model_path="models/KONI-Llama3-8B-20240630.Q4_0.gguf",
                model_path="models/llama-3.2-Korean-Bllossom-3B-Q4_K_M.gguf",
                n_ctx=2048,
                n_threads=8,  # CPU 코어 수에 맞게 증가 (4-16 사이 권장)
                n_threads_batch=8,  # 배치 처리용 스레드도 추가
                n_gpu_layers=0,
                verbose=False,
                n_batch=512,  # 배치 크기 증가
                use_mlock=False,
                use_mmap=True,
                # GPU 설정
                main_gpu=0,  # 주 GPU 설정
                # 성능 최적화 옵션들
                numa=False,  # NUMA 비활성화 (단일 소켓 시스템에서)

            )
            logger.info("모델 로딩 완료")
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            raise e


def create_simple_prompt(message: str) -> str:
    """간단한 프롬프트 생성"""
    return f"다음 질문에 한국어로 간결하고 정확하게 한 번만 답변해주세요.\n\n질문: {message}\n\n답변:"


@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        # 모델 초기화 확인
        if llm is None:
            optimize_system()
            initialize_model()

        data = await request.json()
        user_message = data.get("message", "").strip()

        if not user_message:
            raise HTTPException(status_code=400, detail="메시지가 비어있습니다.")

        logger.info(f"사용자 메시지: {user_message}")

        # 간단한 프롬프트 사용
        prompt = create_simple_prompt(user_message)

        def generate():
            try:
                logger.info(f"프롬프트: {prompt}")  # 디버깅용

                response = llm(
                    prompt,
                    max_tokens=1024,  # 토큰 수 줄여서 반복 방지
                    temperature=0.3,  # 더 낮은 온도로 일관성 증가
                    top_p=0.8,  # 더 보수적 선택
                    repeat_penalty=1.3,  # 반복 페널티 강화
                    frequency_penalty=0.5,  # 빈도 페널티 추가
                    presence_penalty=0.3,  # 존재 페널티 추가
                    echo=False,
                    stream=True,
                    stop=["\n질문:", "질문:", "\n답변:", "답변:", "\n\n", "Human:", "Assistant:", "사용자:", "\n사용자:"]  # 정지 조건 강화
                )

                token_count = 0
                accumulated_text = ""

                for chunk in response:
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        text = chunk['choices'][0].get('text', '')
                        if text:
                            # 반복되는 패턴 감지 및 중단
                            accumulated_text += text

                            # 같은 문장이 반복되는지 확인
                            sentences = accumulated_text.split('.')  
                            if len(sentences) >= 3:
                                last_three = sentences[-3:]
                                if len(set(last_three)) == 1 and last_three[0].strip():  # 같은 문장 반복
                                    logger.info("반복되는 문장 감지, 생성 중단")
                                    break

                            # 질문 형태가 나타나면 중단 (모델이 새로운 질문을 생성하기 시작할 때)
                            if any(marker in text for marker in ["질문:", "Q:", "?"]) and token_count > 10:
                                logger.info("질문 패턴 감지, 생성 중단")
                                break

                            token_count += 1
                            logger.info(f"토큰 #{token_count}: {repr(text)}")  # 디버깅용
                            yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"

                            # 너무 많은 토큰 생성 방지
                            if token_count > 800:
                                logger.info("최대 토큰 수 도달, 생성 중단")
                                break

                logger.info(f"총 생성된 토큰 수: {token_count}")
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"응답 생성 중 오류: {e}")
                error_msg = json.dumps({'error': f'응답 생성 중 오류가 발생했습니다: {str(e)}'}, ensure_ascii=False)
                yield f"data: {error_msg}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"챗 엔드포인트 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def get_chat_interface():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DH-Kkatuli_v1.0</title>
        <meta charset="UTF-8">
        <style>
            #chat-container { 
                max-width: 800px; 
                margin: 20px auto; 
                padding: 20px;
                font-family: 'Malgun Gothic', Arial, sans-serif;
            }
            #messages { 
                margin-bottom: 20px;
                padding: 10px;
                height: 400px;
                overflow-y: auto;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            .message { 
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
                word-wrap: break-word;
            }
            .user { 
                background-color: #e3f2fd;
                margin-left: 20%;
                text-align: right;
            }
            .assistant { 
                background-color: #f5f5f5;
                margin-right: 20%;
            }
            #input-container {
                display: flex;
                gap: 10px;
            }
            #user-input {
                flex-grow: 1;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            button {
                padding: 10px 20px;
                background-color: #2196f3;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:disabled {
                background-color: #ccc;
                cursor: not-allowed;
            }
            #stop-btn {
                background-color: #f44336;
                display: none;
            }
            #stop-btn:hover {
                background-color: #d32f2f;
            }
            .loading {
                color: #999;
                font-style: italic;
            }
            .stopped {
                color: #666;
                font-style: italic;
                border-left: 4px solid #f44336;
            }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <h1>DH-Kkatuli_v1.0</h1>
            <div id="messages"></div>
            <div id="input-container">
                <input type="text" id="user-input" placeholder="메시지를 입력하세요...">
                <button id="send-btn" onclick="sendMessage()">전송</button>
                <button id="stop-btn" onclick="stopGeneration()">중단</button>
            </div>
        </div>
        <script>
            let currentController = null;
            let isGenerating = false;

            function addMessage(role, content, isLoading = false) {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${role}${isLoading ? ' loading' : ''}`;
                messageDiv.textContent = content;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                return messageDiv;
            }

            function updateButtonStates(generating) {
                const sendBtn = document.getElementById('send-btn');
                const stopBtn = document.getElementById('stop-btn');
                const userInput = document.getElementById('user-input');

                isGenerating = generating;
                sendBtn.disabled = generating;
                userInput.disabled = generating;

                if (generating) {
                    stopBtn.style.display = 'block';
                    sendBtn.style.display = 'none';
                } else {
                    stopBtn.style.display = 'none';
                    sendBtn.style.display = 'block';
                }
            }

            function stopGeneration() {
                if (currentController) {
                    currentController.abort();
                    currentController = null;
                }
                updateButtonStates(false);

                // 현재 응답 메시지에 중단됨 표시 추가
                const messages = document.querySelectorAll('.message.assistant');
                if (messages.length > 0) {
                    const lastMessage = messages[messages.length - 1];
                    if (!lastMessage.textContent.includes('[중단됨]')) {
                        lastMessage.textContent += ' [중단됨]';
                        lastMessage.classList.add('stopped');
                    }
                }
            }

            async function sendMessage() {
                const input = document.getElementById('user-input');
                const message = input.value.trim();

                if (!message || isGenerating) return;

                // UI 업데이트
                input.value = '';
                updateButtonStates(true);
                addMessage('user', message);
                const loadingMsg = addMessage('assistant', '응답을 생성하고 있습니다...', true);

                // AbortController 생성
                currentController = new AbortController();

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ message: message }),
                        signal: currentController.signal
                    });

                    if (!response.ok) {
                        throw new Error('서버 오류');
                    }

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let assistantResponse = '';

                    // 로딩 메시지 제거
                    loadingMsg.remove();
                    const responseDiv = addMessage('assistant', '');

                    try {
                        while (true) {
                            const { value, done } = await reader.read();
                            if (done) break;

                            const chunk = decoder.decode(value);
                            const lines = chunk.split('\\n');

                            for (const line of lines) {
                                if (line.startsWith('data: ')) {
                                    const data = line.slice(6);
                                    if (data === '[DONE]') break;

                                    try {
                                        const parsed = JSON.parse(data);
                                        if (parsed.text) {
                                            assistantResponse += parsed.text;
                                            responseDiv.textContent = assistantResponse;
                                        } else if (parsed.error) {
                                            responseDiv.textContent = parsed.error;
                                            responseDiv.className += ' error';
                                            break;
                                        }
                                    } catch (e) {
                                        console.error('JSON 파싱 오류:', e);
                                    }
                                }
                            }
                        }
                    } catch (readError) {
                        if (readError.name === 'AbortError') {
                            console.log('스트림이 사용자에 의해 중단됨');
                        } else {
                            throw readError;
                        }
                    }

                    if (!assistantResponse.trim() && !responseDiv.classList.contains('stopped')) {
                        responseDiv.textContent = '응답을 받지 못했습니다.';
                    }

                } catch (error) {
                    if (error.name === 'AbortError') {
                        console.log('요청이 사용자에 의해 중단됨');
                        if (loadingMsg.parentNode) {
                            loadingMsg.textContent = '요청이 중단되었습니다.';
                            loadingMsg.className = 'message assistant stopped';
                        }
                    } else {
                        console.error('오류:', error);
                        if (loadingMsg.parentNode) {
                            loadingMsg.textContent = '오류가 발생했습니다. 다시 시도해주세요.';
                            loadingMsg.className = 'message assistant error';
                        }
                    }
                } finally {
                    updateButtonStates(false);
                    currentController = null;
                }
            }

            document.getElementById('user-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey && !isGenerating) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            // 페이지 언로드 시 진행 중인 요청 중단
            window.addEventListener('beforeunload', function() {
                if (currentController) {
                    currentController.abort();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
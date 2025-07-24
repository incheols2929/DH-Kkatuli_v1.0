from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from llama_cpp import Llama
import json
import asyncio
import logging
import os
import psutil
import re

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
                model_path="model/EEVE-Korean-Instruct-10.8B-v1.0-Q8_0.gguf",
                # model_path="model/Gugugo-koen-7B-V1.1.Q8_0.gguf",
                # model_path="model/KONI-Llama3-8B-20240630.Q4_0.gguf",
                # model_path="model/llama-3.2-Korean-Bllossom-3B-Q4_K_M.gguf",
                n_ctx=4096,  # 컨텍스트 길이 증가
                n_threads=8,
                n_threads_batch=8,
                n_gpu_layers=0,
                verbose=False,
                n_batch=1024,  # 배치 크기 증가
                use_mlock=False,
                use_mmap=True,
                main_gpu=0,
                numa=False,
                # 추가 성능 최적화 옵션
                seed=-1,  # 랜덤 시드
                flash_attn=True,  # Flash Attention 활성화 (지원되는 경우)
            )
            logger.info("모델 로딩 완료")
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            raise e


def create_optimized_prompt(message: str) -> str:
    """최적화된 프롬프트 생성"""
    # 메시지 타입에 따른 프롬프트 선택
    if any(keyword in message.lower() for keyword in ['코드', 'code', '프로그래밍', '파이썬', '자바']):
        return f"""<|system|>당신은 도움이 되는 AI 어시스턴트입니다. 프로그래밍과 기술 문제에 대해 정확하고 구체적인 답변을 제공합니다.</s>
<|user|>{message}</s>
<|assistant|>"""
    elif any(keyword in message.lower() for keyword in ['설명', '알려줘', '무엇', '어떻게', '순위', '정보']):
        return f"""<|system|>당신은 지식이 풍부한 AI 어시스턴트입니다. 질문에 대해 정확하고 자세한 설명을 제공합니다. 충분한 정보를 포함하여 답변해주세요.</s>
<|user|>{message}</s>
<|assistant|>"""
    else:
        return f"""<|system|>당신은 친근하고 도움이 되는 AI 어시스턴트입니다. 사용자의 질문에 성의껏 답변해주세요.</s>
<|user|>{message}</s>
<|assistant|>"""


def is_response_complete(text: str, token_count: int) -> bool:
    """응답이 완료되었는지 확인 - 더 관대한 조건으로 변경"""
    # 최소 토큰 수 확인 (너무 짧은 답변 방지)
    if token_count < 30:  # 최소 30토큰은 생성하도록
        return False

    # 텍스트 길이도 확인 (한국어 특성상)
    if len(text.strip()) < 50:  # 최소 50자는 생성하도록
        return False

    # 완전한 문장으로 끝나는지 확인 (더 엄격한 패턴)
    complete_patterns = [
        r'.+[.!?]\s*$',  # 내용이 있고 문장부호로 끝남
        r'.+입니다\.$',  # 내용이 있고 "입니다."로 끝남
        r'.+습니다\.$',  # 내용이 있고 "습니다."로 끝남
        r'.+됩니다\.$',  # 내용이 있고 "됩니다."로 끝남
        r'.+있습니다\.$',  # 내용이 있고 "있습니다."로 끝남
        r'.+합니다\.$',  # 내용이 있고 "합니다."로 끝남
    ]

    # 여러 문장이 포함되어 있고 마지막이 완전한 종료인지 확인
    sentences = re.split(r'[.!?]', text.strip())
    if len(sentences) >= 2:  # 최소 2개 문장이 있을 때만 종료 고려
        return any(re.search(pattern, text.strip()) for pattern in complete_patterns)

    return False


def should_stop_generation(accumulated_text: str, current_token: str, token_count: int) -> bool:
    """생성을 중단해야 하는지 판단 - 더 관대하게 수정"""
    # 반복 패턴 감지 (더 엄격하게)
    if token_count > 50:  # 50토큰 이후에만 반복 검사
        words = accumulated_text.split()
        if len(words) >= 15:  # 15개 단어 이상일 때만 검사
            # 최근 7개 단어가 반복되는지 확인 (더 긴 패턴)
            recent_words = words[-7:]
            previous_words = words[-14:-7] if len(words) >= 14 else []
            if recent_words == previous_words and len(set(recent_words)) > 2:
                logger.info(f"반복 패턴 감지: {recent_words}")
                return True

    # 부적절한 패턴 감지
    unwanted_patterns = [
        "질문:",
        "답변:",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
        "Human:",
        "AI:",
        "\n\n질문",
        "\n\n답변",
        "사용자:",
        "\n사용자:"
    ]

    for pattern in unwanted_patterns:
        if pattern in current_token or pattern in accumulated_text[-100:]:  # 마지막 100자에서 검사
            logger.info(f"부적절한 패턴 감지: {pattern}")
            return True

    return False


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

        # 최적화된 프롬프트 사용
        prompt = create_optimized_prompt(user_message)

        def generate():
            try:
                logger.info(f"프롬프트: {prompt[:100]}...")  # 프롬프트 일부만 로깅

                response = llm(
                    prompt,
                    max_tokens=2048,  # 토큰 수 증가
                    temperature=0.7,  # 적절한 창의성
                    top_p=0.9,  # 토큰 선택 범위
                    top_k=40,  # 상위 K개 토큰만 고려
                    repeat_penalty=1.15,  # 반복 페널티
                    frequency_penalty=0.2,  # 빈도 페널티
                    presence_penalty=0.1,  # 존재 페널티
                    echo=False,
                    stream=True,
                    stop=["</s>", "<|user|>", "<|system|>", "Human:", "\n질문:", "\n\n"]  # 일부 stop 조건 완화
                )

                token_count = 0
                accumulated_text = ""

                for chunk in response:
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        text = chunk['choices'][0].get('text', '')
                        if text:
                            accumulated_text += text

                            # 중단 조건 검사
                            if should_stop_generation(accumulated_text, text, token_count):
                                logger.info("부적절한 패턴 감지, 생성 중단")
                                break

                            token_count += 1

                            # 디버깅 로그 레벨 조정
                            if token_count % 20 == 0:  # 20토큰마다 로깅
                                logger.info(f"토큰 #{token_count}: 생성 진행 중... (길이: {len(accumulated_text)})")

                            yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"

                            # 문장 단위로 완료 체크 - 더 관대한 조건
                            if text in '.!?' and is_response_complete(accumulated_text, token_count):
                                logger.info(f"문장 완료 감지, 생성 종료 (토큰: {token_count}, 길이: {len(accumulated_text)})")
                                break

                            # 최대 토큰 수 제한
                            if token_count > 1500:
                                logger.info("최대 토큰 수 도달, 생성 중단")
                                break

                logger.info(f"총 생성된 토큰 수: {token_count}")
                logger.info(f"생성된 텍스트 길이: {len(accumulated_text)}")
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
        <title>DH-Kkatuli_v1.0 (Enhanced)</title>
        <meta charset="UTF-8">
        <style>
            #chat-container { 
                max-width: 900px; 
                margin: 20px auto; 
                padding: 20px;
                font-family: 'Malgun Gothic', Arial, sans-serif;
            }
            #messages { 
                margin-bottom: 20px;
                padding: 15px;
                height: 500px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #fafafa;
            }
            .message { 
                margin: 15px 0;
                padding: 12px 15px;
                border-radius: 8px;
                word-wrap: break-word;
                line-height: 1.5;
            }
            .user { 
                background-color: #e3f2fd;
                margin-left: 15%;
                text-align: right;
                border-left: 4px solid #2196f3;
            }
            .assistant { 
                background-color: #f8f9fa;
                margin-right: 15%;
                border-left: 4px solid #28a745;
            }
            #input-container {
                display: flex;
                gap: 10px;
                margin-top: 15px;
            }
            #user-input {
                flex-grow: 1;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 6px;
                font-size: 14px;
                transition: border-color 0.3s;
            }
            #user-input:focus {
                border-color: #2196f3;
                outline: none;
            }
            button {
                padding: 12px 24px;
                background-color: #2196f3;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #1976d2;
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
                color: #666;
                font-style: italic;
                animation: pulse 1.5s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .stopped {
                color: #666;
                font-style: italic;
                border-left: 4px solid #f44336;
            }
            .error {
                background-color: #ffebee;
                border-left: 4px solid #f44336;
                color: #c62828;
            }
            .stats {
                font-size: 12px;
                color: #666;
                margin-top: 10px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <h1>🤖 DH-Kkatuli_v1.0 (Enhanced)</h1>
            <div class="stats">
                <span>모델: EEVE-Korean-Instruct-10.8B | 최적화된 답변 생성</span>
            </div>
            <div id="messages"></div>
            <div id="input-container">
                <input type="text" id="user-input" placeholder="질문을 입력하세요... (Enter로 전송)">
                <button id="send-btn" onclick="sendMessage()">💬 전송</button>
                <button id="stop-btn" onclick="stopGeneration()">⏹️ 중단</button>
            </div>
        </div>

        <script>
            let currentController = null;
            let isGenerating = false;
            let messageCount = 0;

            function addMessage(role, content, isLoading = false) {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${role}${isLoading ? ' loading' : ''}`;
                messageDiv.textContent = content;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                messageCount++;
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
                    stopBtn.style.display = 'inline-block';
                    sendBtn.style.display = 'none';
                } else {
                    stopBtn.style.display = 'none';
                    sendBtn.style.display = 'inline-block';
                }
            }

            function stopGeneration() {
                if (currentController) {
                    currentController.abort();
                    currentController = null;
                }
                updateButtonStates(false);

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

                input.value = '';
                updateButtonStates(true);
                addMessage('user', message);
                const loadingMsg = addMessage('assistant', '🤔 생각하는 중...', true);

                currentController = new AbortController();
                const startTime = Date.now();

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
                        throw new Error(`서버 오류: ${response.status}`);
                    }

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let assistantResponse = '';

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
                                    if (data === '[DONE]') {
                                        const duration = ((Date.now() - startTime) / 1000).toFixed(1);
                                        console.log(`응답 완료 (${duration}초)`);
                                        break;
                                    }

                                    try {
                                        const parsed = JSON.parse(data);
                                        if (parsed.text) {
                                            assistantResponse += parsed.text;
                                            responseDiv.textContent = assistantResponse;
                                            responseDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });
                                        } else if (parsed.error) {
                                            responseDiv.textContent = `❌ ${parsed.error}`;
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
                        responseDiv.textContent = '❌ 응답을 받지 못했습니다.';
                        responseDiv.className += ' error';
                    }

                } catch (error) {
                    if (error.name === 'AbortError') {
                        console.log('요청이 사용자에 의해 중단됨');
                        if (loadingMsg.parentNode) {
                            loadingMsg.textContent = '⏹️ 요청이 중단되었습니다.';
                            loadingMsg.className = 'message assistant stopped';
                        }
                    } else {
                        console.error('오류:', error);
                        if (loadingMsg.parentNode) {
                            loadingMsg.textContent = `❌ 오류: ${error.message}`;
                            loadingMsg.className = 'message assistant error';
                        }
                    }
                } finally {
                    updateButtonStates(false);
                    currentController = null;
                }
            }

            // 엔터 키 이벤트
            document.getElementById('user-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey && !isGenerating) {
                    e.preventDefault();
                    sendMessage();
                }
            });

            // 페이지 언로드 시 정리
            window.addEventListener('beforeunload', function() {
                if (currentController) {
                    currentController.abort();
                }
            });

            // 초기 포커스
            document.getElementById('user-input').focus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="192.168.0.13", port=8000)
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from llama_cpp import Llama
import json
import asyncio
import logging
import os
import psutil
import re
from datetime import datetime
import traceback
from contextlib import asynccontextmanager
from typing import List, Dict


# 로깅 설정 강화
def setup_logging():
    """향상된 로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 현재 날짜로 로그 파일명 생성
    log_filename = f"{log_dir}/kkatuli_{datetime.now().strftime('%Y%m%d')}.log"

    # 로깅 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 파일 핸들러 (모든 로그를 파일에 저장)
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # 콘솔 핸들러 (INFO 이상만 콘솔에 출력)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 핸들러가 중복되지 않도록 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 새 핸들러 추가
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)


# 로깅 설정 초기화
logger = setup_logging()

# 모델을 전역 변수로 선언하되 초기화는 지연
llm = None

# 대화 기록을 저장할 전역 딕셔너리 (세션 ID별로 관리)
conversation_history: Dict[str, List[Dict[str, str]]] = {}

# 성능 통계를 위한 글로벌 변수
performance_stats = {
    "total_requests": 0,
    "total_tokens_generated": 0,
    "total_processing_time": 0,
    "error_count": 0
}


def log_performance_stats():
    """성능 통계 로깅"""
    logger.info("=" * 50)
    logger.info("성능 통계 요약:")
    logger.info(f"총 요청 수: {performance_stats['total_requests']}")
    logger.info(f"총 생성된 토큰 수: {performance_stats['total_tokens_generated']}")
    logger.info(f"총 처리 시간: {performance_stats['total_processing_time']:.2f}초")
    logger.info(
        f"평균 처리 시간: {performance_stats['total_processing_time'] / max(1, performance_stats['total_requests']):.2f}초/요청")
    logger.info(
        f"평균 토큰 생성 속도: {performance_stats['total_tokens_generated'] / max(1, performance_stats['total_processing_time']):.2f}토큰/초")
    logger.info(f"오류 발생 수: {performance_stats['error_count']}")
    logger.info("=" * 50)


def optimize_system():
    """시스템 최적화 설정"""
    logger.info("시스템 최적화 시작...")
    try:
        # CPU 정보 로깅
        cpu_count = psutil.cpu_count()
        memory_info = psutil.virtual_memory()
        logger.info(f"CPU 코어 수: {cpu_count}")
        logger.info(f"전체 메모리: {memory_info.total / (1024 ** 3):.1f}GB")
        logger.info(f"사용 가능한 메모리: {memory_info.available / (1024 ** 3):.1f}GB")

        # CPU 친화도 설정 (모든 코어 사용)
        process = psutil.Process()
        process.cpu_affinity(list(range(cpu_count)))
        logger.info(f"CPU 친화도 설정 완료: 모든 {cpu_count}개 코어 사용")

        # 우선순위 높음으로 설정
        process.nice(psutil.HIGH_PRIORITY_CLASS)
        logger.info("프로세스 우선순위를 높음으로 설정 완료")

    except Exception as e:
        logger.error(f"시스템 최적화 중 오류 발생: {e}")
        logger.error(traceback.format_exc())


def initialize_model():
    """모델을 안전하게 초기화"""
    global llm
    if llm is None:
        try:
            logger.info("=" * 50)
            logger.info("모델 로딩 시작...")
            start_time = datetime.now()

            model_path = "model/EEVE-Korean-Instruct-10.8B-v1.0-Q8_0.gguf"
            logger.info(f"모델 파일: {model_path}")

            # 모델 파일 존재 확인
            if not os.path.exists(model_path):
                logger.error(f"모델 파일을 찾을 수 없습니다: {model_path}")
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

            # 모델 파일 크기 확인
            model_size = os.path.getsize(model_path) / (1024 ** 3)
            logger.info(f"모델 파일 크기: {model_size:.1f}GB")

            llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=8,
                n_threads_batch=8,
                n_gpu_layers=0,
                verbose=False,
                n_batch=1024,
                use_mlock=False,
                use_mmap=True,
                main_gpu=0,
                numa=False,
                seed=-1,
                flash_attn=True,
            )

            end_time = datetime.now()
            loading_time = (end_time - start_time).total_seconds()
            logger.info(f"모델 로딩 완료 (소요 시간: {loading_time:.2f}초)")
            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            logger.error(traceback.format_exc())
            performance_stats["error_count"] += 1
            raise e


def get_conversation_context(session_id: str, max_turns: int = 5) -> str:
    """대화 기록에서 컨텍스트를 생성"""
    if session_id not in conversation_history:
        return ""

    history = conversation_history[session_id]
    # 최근 max_turns 개의 대화만 포함
    recent_history = history[-max_turns * 2:] if len(history) > max_turns * 2 else history

    context = ""
    for entry in recent_history:
        if entry["role"] == "user":
            context += f"사용자: {entry['content']}\n"
        elif entry["role"] == "assistant":
            context += f"AI: {entry['content']}\n"

    return context


def add_to_conversation_history(session_id: str, role: str, content: str):
    """대화 기록에 추가"""
    if session_id not in conversation_history:
        conversation_history[session_id] = []

    conversation_history[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

    # 대화 기록이 너무 길어지면 오래된 것들 제거 (최대 20개 유지)
    if len(conversation_history[session_id]) > 20:
        conversation_history[session_id] = conversation_history[session_id][-20:]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # 시작 시 실행
    logger.info("=" * 60)
    logger.info("DH-Kkatuli_v1.0 서버 시작")
    logger.info(f"시작 시간: {datetime.now()}")
    logger.info("=" * 60)

    yield  # 애플리케이션 실행

    # 종료 시 실행
    logger.info("=" * 60)
    logger.info("DH-Kkatuli_v1.0 서버 종료")
    logger.info(f"종료 시간: {datetime.now()}")
    log_performance_stats()
    logger.info("=" * 60)


# FastAPI 앱 생성 시 lifespan 설정
app = FastAPI(lifespan=lifespan)


def create_optimized_prompt(message: str, context: str = "") -> str:
    """최적화된 프롬프트 생성 - 대화 컨텍스트 포함"""
    logger.debug(f"프롬프트 생성 중 - 메시지 길이: {len(message)}, 컨텍스트 길이: {len(context)}")

    # 기본 시스템 메시지
    system_msg = """당신은 도움이 되는 AI 어시스턴트입니다. 사용자와의 이전 대화 내용을 참고하여 일관성 있고 연속적인 답변을 제공합니다. 
이전 대화에서 언급된 내용이 있다면 그것을 기반으로 답변을 이어가고, 필요시 추가적인 정보나 상세한 설명을 제공해주세요. 
가능한 한 자세하고 완전한 답변을 해주세요."""

    if any(keyword in message.lower() for keyword in ['코드', 'code', '프로그래밍', '파이썬', '자바']):
        system_msg += " 프로그래밍과 기술 문제에 대해 정확하고 구체적인 답변을 제공합니다."
        logger.debug("코딩 관련 프롬프트 생성")
    elif any(keyword in message.lower() for keyword in ['설명', '알려줘', '무엇', '어떻게', '순위', '정보']):
        system_msg += " 질문에 대해 정확하고 자세한 설명을 제공합니다. 충분한 정보와 배경 지식을 포함하여 완전한 답변을 해주세요."
        logger.debug("설명 관련 프롬프트 생성")

    # 대화 컨텍스트가 있는 경우 포함
    if context.strip():
        prompt = f"""<|system|>{system_msg}

이전 대화 내용:
{context}</s>
<|user|>{message}</s>
<|assistant|>"""
    else:
        prompt = f"""<|system|>{system_msg}</s>
<|user|>{message}</s>
<|assistant|>"""

    return prompt


def is_response_complete(text: str, token_count: int) -> bool:
    """응답이 완료되었는지 확인 - 더 엄격한 조건으로 변경"""
    logger.debug(f"응답 완료 검사 - 토큰: {token_count}, 텍스트 길이: {len(text)}")

    # 최소 토큰 수 확인 (더 많은 토큰 요구)
    if token_count < 100:
        logger.debug("최소 토큰 수 미달")
        return False

    # 텍스트 길이도 확인 (한국어 특성상)
    if len(text.strip()) < 200:
        logger.debug("최소 텍스트 길이 미달")
        return False

    # 완전한 문장으로 끝나는지 확인
    complete_patterns = [
        r'.+[.!?]\s*$',
        r'.+입니다\.$',
        r'.+습니다\.$',
        r'.+됩니다\.$',
        r'.+있습니다\.$',
        r'.+합니다\.$',
    ]

    # 충분한 내용과 문장이 포함되어 있는지 확인
    sentences = re.split(r'[.!?]', text.strip())
    if len(sentences) >= 3:
        is_complete = any(re.search(pattern, text.strip()) for pattern in complete_patterns)
        logger.debug(f"문장 완료 검사 결과: {is_complete}")
        return is_complete

    return False


def should_stop_generation(accumulated_text: str, current_token: str, token_count: int) -> bool:
    """생성을 중단해야 하는지 판단 - 더 관대하게 수정"""
    # 반복 패턴 감지 (더 엄격하게)
    if token_count > 50:
        words = accumulated_text.split()
        if len(words) >= 15:
            recent_words = words[-7:]
            previous_words = words[-14:-7] if len(words) >= 14 else []
            if recent_words == previous_words and len(set(recent_words)) > 2:
                logger.warning(f"반복 패턴 감지하여 생성 중단: {recent_words}")
                return True

    # 부적절한 패턴 감지
    unwanted_patterns = [
        "질문:", "답변:", "<|user|>", "<|assistant|>", "<|system|>",
        "Human:", "AI:", "\n\n질문", "\n\n답변", "사용자:", "\n사용자:"
    ]

    for pattern in unwanted_patterns:
        if pattern in current_token or pattern in accumulated_text[-100:]:
            logger.warning(f"부적절한 패턴 감지하여 생성 중단: {pattern}")
            return True

    return False


@app.post("/chat")
async def chat_endpoint(request: Request):
    request_start_time = datetime.now()
    request_id = f"REQ_{request_start_time.strftime('%Y%m%d_%H%M%S_%f')}"

    logger.info(f"[{request_id}] 새로운 채팅 요청 시작")
    performance_stats["total_requests"] += 1

    try:
        # 클라이언트 IP 로깅
        client_ip = request.client.host
        logger.info(f"[{request_id}] 클라이언트 IP: {client_ip}")

        # 모델 초기화 확인
        if llm is None:
            logger.info(f"[{request_id}] 모델이 초기화되지 않음, 초기화 시작")
            optimize_system()
            initialize_model()

        data = await request.json()
        user_message = data.get("message", "").strip()
        session_id = data.get("session_id", "default")  # 세션 ID 추가
        continue_conversation = data.get("continue", False)  # 대화 이어가기 플래그

        if not user_message:
            logger.warning(f"[{request_id}] 빈 메시지 요청")
            raise HTTPException(status_code=400, detail="메시지가 비어있습니다.")

        logger.info(f"[{request_id}] 사용자 메시지: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
        logger.info(f"[{request_id}] 세션 ID: {session_id}, 대화 이어가기: {continue_conversation}")

        # 대화 기록에 사용자 메시지 추가
        add_to_conversation_history(session_id, "user", user_message)

        # 대화 컨텍스트 생성
        context = ""
        if continue_conversation or session_id in conversation_history:
            context = get_conversation_context(session_id)
            logger.info(f"[{request_id}] 대화 컨텍스트 길이: {len(context)} 문자")

        # 최적화된 프롬프트 사용 (컨텍스트 포함)
        prompt = create_optimized_prompt(user_message, context)
        logger.debug(f"[{request_id}] 생성된 프롬프트 길이: {len(prompt)}")

        def generate():
            generation_start_time = datetime.now()
            token_count = 0
            accumulated_text = ""

            try:
                logger.info(f"[{request_id}] 텍스트 생성 시작")

                response = llm(
                    prompt,
                    max_tokens=3072,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=50,
                    repeat_penalty=1.1,
                    frequency_penalty=0.1,
                    presence_penalty=0.05,
                    echo=False,
                    stream=True,
                    stop=["</s>", "<|user|>", "<|system|>"]
                )

                for chunk in response:
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        text = chunk['choices'][0].get('text', '')
                        if text:
                            accumulated_text += text
                            token_count += 1

                            # 중단 조건 검사
                            if should_stop_generation(accumulated_text, text, token_count):
                                logger.warning(f"[{request_id}] 부적절한 패턴 감지로 생성 중단")
                                break

                            # 주기적 로깅
                            if token_count % 50 == 0:
                                logger.info(f"[{request_id}] 진행 상황 - 토큰: {token_count}, 길이: {len(accumulated_text)}")

                            yield f"data: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"

                            # 문장 단위로 완료 체크
                            if text in '.!?' and is_response_complete(accumulated_text, token_count):
                                logger.info(f"[{request_id}] 문장 완료 감지로 생성 종료")
                                break

                            # 최대 토큰 수 제한
                            if token_count > 1500:
                                logger.info(f"[{request_id}] 최대 토큰 수 도달로 생성 중단")
                                break

                # 대화 기록에 AI 응답 추가
                if accumulated_text.strip():
                    add_to_conversation_history(session_id, "assistant", accumulated_text.strip())

                generation_end_time = datetime.now()
                generation_time = (generation_end_time - generation_start_time).total_seconds()

                # 성능 통계 업데이트
                performance_stats["total_tokens_generated"] += token_count
                performance_stats["total_processing_time"] += generation_time

                logger.info(
                    f"[{request_id}] 생성 완료 - 토큰: {token_count}, 시간: {generation_time:.2f}초, 속도: {token_count / max(generation_time, 0.001):.2f}토큰/초")
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"[{request_id}] 응답 생성 중 오류: {e}")
                logger.error(f"[{request_id}] 오류 상세: {traceback.format_exc()}")
                performance_stats["error_count"] += 1
                error_msg = json.dumps({'error': f'응답 생성 중 오류가 발생했습니다: {str(e)}'}, ensure_ascii=False)
                yield f"data: {error_msg}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"[{request_id}] 챗 엔드포인트 오류: {e}")
        logger.error(f"[{request_id}] 오류 상세: {traceback.format_exc()}")
        performance_stats["error_count"] += 1
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        request_end_time = datetime.now()
        total_request_time = (request_end_time - request_start_time).total_seconds()
        logger.info(f"[{request_id}] 요청 완료 - 총 소요 시간: {total_request_time:.2f}초")

        # 매 10번째 요청마다 성능 통계 출력
        if performance_stats["total_requests"] % 10 == 0:
            log_performance_stats()


@app.get("/history/{session_id}")
async def get_conversation_history(session_id: str):
    """특정 세션의 대화 기록 조회"""
    logger.info(f"대화 기록 조회 요청 - 세션 ID: {session_id}")
    if session_id in conversation_history:
        return {"session_id": session_id, "history": conversation_history[session_id]}
    return {"session_id": session_id, "history": []}


@app.delete("/history/{session_id}")
async def clear_conversation_history(session_id: str):
    """특정 세션의 대화 기록 초기화"""
    logger.info(f"대화 기록 초기화 요청 - 세션 ID: {session_id}")
    if session_id in conversation_history:
        del conversation_history[session_id]
        return {"message": f"세션 {session_id}의 대화 기록이 초기화되었습니다."}
    return {"message": f"세션 {session_id}의 대화 기록이 존재하지 않습니다."}


@app.get("/")
async def get_chat_interface():
    logger.info("웹 인터페이스 요청")
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DH-Kkatuli_v1.0 (대화 연속 기능 추가)</title>
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
            #control-container {
                display: flex;
                gap: 10px;
                margin-bottom: 10px;
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
            #continue-btn {
                background-color: #ff9800;
            }
            #continue-btn:hover {
                background-color: #f57c00;
            }
            #clear-btn {
                background-color: #9e9e9e;
            }
            #clear-btn:hover {
                background-color: #757575;
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
            .session-info {
                font-size: 12px;
                color: #666;
                margin-bottom: 10px;
                padding: 8px;
                background-color: #f5f5f5;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <h1>🤖 DH-Kkatuli_v1.0 (대화 연속 기능 추가)</h1>
            <div class="stats">
                <span>모델: EEVE-Korean-Instruct-10.8B | 대화 기록 유지 | 연속 답변 기능</span>
            </div>
            <div class="session-info">
                세션 ID: <span id="session-id">default</span> | 
                대화 수: <span id="message-count">0</span>
            </div>
            <div id="control-container">
                <button id="continue-btn" onclick="continueConversation()">🔄 대화 이어가기</button>
                <button id="clear-btn" onclick="clearHistory()">🗑️ 대화 기록 지우기</button>
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
            let sessionId = 'default';

            function addMessage(role, content, isLoading = false) {
                const messagesDiv = document.getElementById('messages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${role}${isLoading ? ' loading' : ''}`;
                messageDiv.textContent = content;
                messagesDiv.appendChild(messageDiv);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                messageCount++;
                updateMessageCount();
                return messageDiv;
            }

            function updateMessageCount() {
                document.getElementById('message-count').textContent = Math.floor(messageCount / 2);
            }

            function updateButtonStates(generating) {
                const sendBtn = document.getElementById('send-btn');
                const stopBtn = document.getElementById('stop-btn');
                const userInput = document.getElementById('user-input');
                const continueBtn = document.getElementById('continue-btn');
                const clearBtn = document.getElementById('clear-btn');

                isGenerating = generating;
                sendBtn.disabled = generating;
                userInput.disabled = generating;
                continueBtn.disabled = generating;
                clearBtn.disabled = generating;

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

            async function sendMessage(continueMode = false) {
                const input = document.getElementById('user-input');
                let message = input.value.trim();

                if (continueMode) {
                    // 대화 이어가기 모드에서는 기본 메시지 사용
                    message = message || "위의 답변에 이어서 더 자세히 설명해주세요.";
                }

                if (!message || isGenerating) return;

                input.value = '';
                updateButtonStates(true);

                if (!continueMode) {
                    addMessage('user', message);
                }

                const loadingMsg = addMessage('assistant', '🤔 생각하는 중...', true);

                currentController = new AbortController();
                const startTime = Date.now();

                try {
                    const requestBody = {
                        message: message,
                        session_id: sessionId,
                        continue: continueMode
                    };

                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(requestBody),
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

            function continueConversation() {
                sendMessage(true);
            }

            async function clearHistory() {
                if (confirm('대화 기록을 모두 지우시겠습니까?')) {
                    try {
                        await fetch(`/history/${sessionId}`, {
                            method: 'DELETE'
                        });
                        document.getElementById('messages').innerHTML = '';
                        messageCount = 0;
                        updateMessageCount();
                        console.log('대화 기록이 초기화되었습니다.');
                    } catch (error) {
                        console.error('대화 기록 초기화 오류:', error);
                    }
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

    logger.info("서버 시작 준비 중...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
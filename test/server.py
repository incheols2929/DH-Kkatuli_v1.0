from fastapi import FastAPI
from langserve import add_routes
from chain import chain as myChain

from typing import List, Union
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class InputChat(BaseModel):
    messsages1: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)  # 이 줄 추가


app = FastAPI()
add_routes(
    app,
    myChain.with_types(input_type=InputChat),
    path="/ollama",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)

#http://localhost:8000/ollama/playground/
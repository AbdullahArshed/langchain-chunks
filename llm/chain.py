from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY

def get_chain(memory, streaming=False, callbacks=None):
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=OPENAI_API_KEY,
        streaming=streaming,
        callbacks=callbacks,
    )
    return ConversationChain(llm=llm, memory=memory, verbose=True)

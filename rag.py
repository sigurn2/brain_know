from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import asyncio
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from shutil import move

# configuration
model_name = "/root/brain_know/bge-m3"
model_kwargs = {"device": "auto"}
embedding = HuggingFaceEmbeddings(model_name=model_name)


def metadata_func(input: dict, output: dict) -> dict:
    output["PMID "] = input.get("PMID")
    return output


def load_data(file_path: str):
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.[] | {PMID: .PMID, title: .title, abstract: .abstract}',
        text_content=False,
        metadata_func=metadata_func
    )
    data = loader.load()
    return data


def get_db(vector_db_path="vectordb") -> FAISS:
    count = filter()
    print(f"!!!!!!!!!!!!!! found {count} files in info")
    if os.path.isdir(vector_db_path) and count == 0:
        print(">>>>>>>>>>>>>> found existd full db <<<<<<<<<<<<<<<<<")
        db = FAISS.load_local(vector_db_path, embeddings=embedding, allow_dangerous_deserialization=True)
        return db
    elif count > 0 and os.path.isdir(vector_db_path):
        print(">>>>>>>>>>>>>> found new files <<<<<<<<<<<<<<<<<")
        db = FAISS.load_local(vector_db_path, embeddings=embedding, allow_dangerous_deserialization=True)
        asyncio.run(update_vectordb(db))
        return db
    else:
        print(">>>>>>>>>>>>>> start build a db <<<<<<<<<<<<<<<")
        return asyncio.run(init_vectordb())


def filter(folder_path="info/"):
    count = 0
    print("!!!!!!!!!!!!!!!!start validating !!!!!!!!!!!!!!!!!!!!!")
    for root, _, files in os.walk(folder_path, topdown=False):
        for file_name in files:
            path = os.path.join(root, file_name)
            try:
                fd = open(path, 'r')
                assert len(fd.read()) > 2
                fd.close()
                count += 1
            except:
                move(path, "garbage")
    return count


async def add(db, data, name):
    db.add_documents(data)
    move(name, "flagged")


async def update_vectordb(db: FAISS, folder_path="info/", vector_db_path="vectordb") -> FAISS:
    file_names = []
    for root, _, files in os.walk(folder_path, topdown=False):
        for name in files:
            file_names.append(os.path.join(root, name))
    print(">>>>>>>>>>>found files <<<<<<<<<<<<")
    for i in range(0, len(file_names)):
        try:
            data = load_data(file_names[i])
            print(">>>>>>in progress <<<<<<")
            asyncio.create_task(db.aadd_documents(data))
            move(file_names[i], "flagged")
            print(">>>>>>finish <<<<<<<<<<")
            db.save_local("vectordb")
        except:
            continue


# init a vector database 
def init_vectordb(folder_path="info/") -> FAISS:
    # save data into db
    # get all json file 
    file_names = []
    for root, _, files in os.walk(folder_path, topdown=False):
        for name in files:
            file_names.append(os.path.join(root, name))
    print(">>>>>>>>>>>found files <<<<<<<<<<<<")
    db = FAISS
    try:
        data = load_data(file_names[0])
        print(">>>>>>>>initializing vector db <<<<<<<<<<<<<")
        db = FAISS.from_documents(data, embedding=embedding)
        db.save_local("vectordb")
        move(file_names[0],"flagged")
        print(">>>>>>>>>build succeed <<<<<<<<")
    except:
        print(">>>>>>>first file corrupted! NEED TO DELETE THIS FILE<<<<<<<<<<<<<<<<<<<<<")
        
    db = FAISS.load_local("vectordb", embeddings=embedding, allow_dangerous_deserialization=True)
    asyncio.run(update_vectordb(db))
    print(">>>>>>>>>build succeed <<<<<<<<")
    return db


async def apppend_db(sem, db: FAISS, data, file_name):
    backgroud_tasks = []
    async with sem:
        task = asyncio.create_task(add(db=db, data=data, file_name=file_name))
        backgroud_tasks.append(task)
    await asyncio.wait(backgroud_tasks)
    return db


if __name__ == "__main__":
    # use db as retriever,set k documents of top = 1 threshold=0.8 
    db = get_db()
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 1, "score_threshold": 0.8})


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    # define a llm
    from langchain_openai import ChatOpenAI

    os.environ['OPENAI_API_KEY'] = "sk-abdasdfasdfasdfasdfx"
    llm = ChatOpenAI(model="gpt-3.5-turbo", base_url=f'aasdfsdf')
    # build langchain
    prompt = PromptTemplate.from_template(
        """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:"""
    )
    # define a chain
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    chat_history = []


    def chat(message):
        documents = retriever.invoke(message)
        answer = rag_chain.invoke(message)
        return answer, documents


    # rag_chain.invoke("Conformation")
    import gradio as gr

    gr.Interface(
        fn=chat,
        inputs="text",
        outputs=["text"],
        title=f"Brain know",
        description="Alright reserved by BrainCog Lab",
    ).launch()

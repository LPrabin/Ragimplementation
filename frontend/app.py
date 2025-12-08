import time
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Notebook", layout="wide")

st.title("Notebook")

# Sidebar for Notebook Selection
st.sidebar.header("Notebooks")
#have dedicated chat for every notebook

# Fetch notebooks
try:
    notebooks = requests.get(f"{API_URL}/notebooks").json()
except requests.exceptions.ConnectionError:
    st.error("Backend is starting... please wait.")
    time.sleep(5)
    st.rerun()

selected_notebook = st.sidebar.selectbox("Select Notebook", notebooks)

new_notebook_name = st.sidebar.text_input("New Notebook Name")
if st.sidebar.button("Create Notebook"):
    if new_notebook_name:
        requests.post(f"{API_URL}/notebooks", json={"name": new_notebook_name})
        st.rerun()

if selected_notebook:
    if st.sidebar.button("Delete Notebook"):
        requests.post(f"{API_URL}/notebooks/delete", json={"name": selected_notebook})
        st.rerun()

    st.header(f"Notebook: {selected_notebook}")
    
    tab1, tab2 = st.tabs(["Chat", "Sources"])
    
    with tab2:
        st.subheader("Manage Sources")
        
        # Chunking Configuration
        col_c1, col_c2 = st.columns(2)
        chunk_size = col_c1.number_input("Chunk Size", min_value=100, max_value=5000, value=1000)
        chunk_overlap = col_c2.number_input("Chunk Overlap", min_value=0, max_value=1000, value=100)
        
        uploaded_files = st.file_uploader("Add Source (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)
        if uploaded_files:
            if st.button("Upload"):
                files_payload = [("files", (file.name, file, file.type)) for file in uploaded_files]
                data_payload = {
                    "notebook_name": selected_notebook, 
                    "chunk_size": chunk_size, 
                    "chunk_overlap": chunk_overlap
                }
                
                with st.spinner("Starting ingestion..."):
                    res = requests.post(f"{API_URL}/resources/add", data=data_payload, files=files_payload)
                    if res.status_code == 200:
                        data = res.json()
                        task_id = data["task_id"]
                        st.success(f"Ingestion started! Task ID: {task_id}")
                        st.session_state["last_task_id"] = task_id # Store for checking
                    else:
                        st.error(f"Failed to upload: {res.text}")
        
        # Status Checker
        if "last_task_id" in st.session_state:
            st.divider()
            st.write(f"Last Task ID: `{st.session_state['last_task_id']}`")
            if st.button("Check Status"):
                try:
                    status_res = requests.get(f"{API_URL}/ingestion/task_status", params={"task_id": st.session_state["last_task_id"]})
                    if status_res.status_code == 200:
                        st.json(status_res.json())
                    else:
                        st.error(f"Error checking status: {status_res.text}")
                except Exception as e:
                    st.error(str(e))

        st.divider()
        st.subheader("Existing Sources")
        resources = requests.post(f"{API_URL}/resources/list", json={"name": selected_notebook}).json()
        for res in resources:
            col1, col2 = st.columns([4, 1])
            col1.text(res)
            if col2.button("Delete", key=res):
                requests.post(f"{API_URL}/resources/delete", json={"notebook_name": selected_notebook, "resource_name": res})
                st.rerun()

    with tab1:
        st.subheader("Chat with your Notebook")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                response = requests.post(f"{API_URL}/query", json={"notebook_name": selected_notebook, "query": prompt})
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = list(set(data["sources"]))
                    top3docs_data = data.get("top3docs", []) 
                    top3docs = list(top3docs_data)
                    
                    
                    
                    docs_content = "\n\n".join([f"**Source {i+1}:**\n> {doc}" for i, doc in enumerate(top3docs)])
                    full_response = f"{answer}\n\n**Sources:** {', '.join(sources)}\n\n**Retrieved Context:**\n\n{docs_content}"
                    
                    with st.chat_message("assistant"):
                        st.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("Error getting response.")

else:
    st.info("Select or create a notebook to get started.")

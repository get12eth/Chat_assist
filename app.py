import os
import tempfile
from pathlib import Path
import pandas as pd
import panel as pn
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import gc # Import garbage collector

# Initialize Panel
pn.extension(design='material')

# --- Configuration ---
SUPPORTED_FILE_TYPES = ['.pdf', '.docx']
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODELS = {
    "FLAN-T5 Small": "google/flan-t5-small",
    "BART-MNLI": "facebook/bart-large-mnli",
    "DistilGPT2": "distilgpt2"
}
# Directory to persist the Chroma database
PERSIST_DIRECTORY = "chroma_db_persist_multi"

# --- Custom CSS (keep as is) ---
css = """
.bk-root .title {
    font-size: 1.5em !important;
    font-weight: bold !important;
    margin-bottom: 20px !important;
}
.bk-root .chat-box {
    min-height: 500px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.bk-root .sidebar {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
}
.file-display {
    max-height: 200px;
    overflow-y: auto;
    margin-top: 10px;
    border: 1px solid #ddd;
    padding: 10px;
    border-radius: 5px;
}
.file-item {
    padding: 5px;
    border-bottom: 1px solid #eee;
}
.file-item:last-child {
    border-bottom: none;
}
"""
pn.config.raw_css = [css]

# --- Global State ---
# Initialize embeddings model once
print("Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print("Embedding model initialized.")

# Initialize the single Chroma database (load if exists, create if not)
print(f"Initializing Chroma DB from/to: {PERSIST_DIRECTORY}")
# Ensure the directory exists if we want to persist
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
main_db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)
print(f"Chroma DB initialized. Number of existing documents: {main_db._collection.count()}")

# Global retriever variable (initially None or based on existing DB)
main_retriever = None
if main_db._collection.count() > 0:
     # If DB already had items, create a retriever for them immediately
     # Use a default 'k' value initially, can be updated later
     print("Creating retriever for existing documents in DB...")
     main_retriever = main_db.as_retriever(search_kwargs={"k": 2}) # Default k=2
     print("Retriever created.")
else:
    print("DB is empty, retriever will be created after first document load.")


# Keep track of the initial empty DataFrame structure for uploaded_files
initial_df_columns = ["File Name", "Size (KB)", "Type", "Chunks Added"]
initial_empty_df = pd.DataFrame(columns=initial_df_columns)

# --- UI Components ---
# Try to load existing file list if DB wasn't empty (more advanced state persistence)
# For simplicity now, we'll just rely on the DB count and rebuild the UI list on load.
# A more robust solution would store the file list metadata separately.
uploaded_files = pn.widgets.DataFrame(initial_empty_df.copy(), name="Loaded Documents", height=150, disabled=True)

file_input = pn.widgets.FileInput(
    accept=','.join(SUPPORTED_FILE_TYPES),
    name="Select Document (PDF or Word)",
    height=50,
    multiple=False # Keep as False, process one at a time
)

upload_btn = pn.widgets.Button(
    name="Load Document",
    button_type="primary",
    disabled=True,
    width=140
)

file_status_indicator = pn.pane.HTML(
    "",
    width=30,
    height=30,
    align='center',
    margin=(10, 0, 0, 5)
)

model_select = pn.widgets.Select(
    name="LLM Model",
    options=list(LLM_MODELS.keys()),
    value="FLAN-T5 Small"
)

k_slider = pn.widgets.IntSlider(
    name="Top K Chunks (Overall)", # Renamed label
    start=1,
    end=10, # Increased max k, might need more with multiple docs
    step=1,
    value=2 # Default k
)

chain_select = pn.widgets.RadioButtonGroup(
    name="Processing Method",
    options=["stuff", "map_reduce", "refine", "map_rerank"],
    button_type="success",
    value="stuff"
)

chat_input = pn.widgets.TextInput(
    placeholder="Load documents first...",
    disabled=True # Disabled until a retriever is ready
)

submit_btn = pn.widgets.Button(
    name="Ask Question",
    button_type="primary",
    disabled=True, # Disabled until a retriever is ready
    width=200
)

clear_btn = pn.widgets.Button(
    name="Clear Chat",
    button_type="warning",
    width=200
)

# Optional: Button to clear the persistent database and UI list
clear_db_btn = pn.widgets.Button(
    name="Clear All Loaded Data",
    button_type="danger",
    width=200,
    margin=(0, 5) # Add some margin
)


processing_spinner = pn.indicators.LoadingSpinner(
    value=False,
    size=25,
    color="primary",
    name="Processing..."
)

# --- Document Processing Functions ---

def load_document(file_path):
    """Load either PDF or Word document and add filename metadata"""
    ext = Path(file_path).suffix.lower()
    print(f"Loading document: {file_path} with extension {ext}")
    if ext == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext == '.docx':
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    file_name = Path(file_path).name
    for doc in docs:
        # Ensure metadata exists
        if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
            doc.metadata = {}
        doc.metadata["filename"] = file_name # Store filename in metadata
        # Add page number if available (PyPDFLoader adds it)
        if 'page' not in doc.metadata:
             doc.metadata['page'] = doc.metadata.get('page_number', 'N/A') # Handle potential variations

    print(f"Loaded {len(docs)} pages/sections from {file_name}")
    return docs

# This function is no longer needed as we don't create separate retrievers
# def create_retriever(...):
#     pass

# --- Callbacks and Interactivity ---

def update_upload_button(event):
    """Enable/disable upload button based on file selection"""
    upload_btn.disabled = not bool(file_input.value)

# REFACTORED: Adds docs to the single persistent DB
def process_uploaded_file(event):
    """Handle file processing: load, split, add to main DB, update retriever."""
    if not file_input.value or not file_input.filename:
        print("Upload button clicked but no file selected.")
        return

    global main_db, main_retriever # We modify these global objects

    file_content = file_input.value
    file_name = file_input.filename
    temp_path = None

    # --- Check for Duplicates ---
    current_files_df = uploaded_files.value
    if file_name in current_files_df["File Name"].values:
        file_status_indicator.object = '<span style="font-size: 1.5em; color: orange;">ℹ️</span>' # Info icon
        print(f"File '{file_name}' is already loaded according to the list. Skipping.")
        # Update placeholder if needed, but don't re-process
        if main_retriever and chat_input.disabled:
             chat_input.disabled = False
             chat_input.placeholder = "Ask questions about loaded documents..."
             submit_btn.disabled = False
        file_input.value = None # Clear input widget
        upload_btn.disabled = True # Disable until new file selected
        return
    # --- End Duplicate Check ---

    processing_spinner.value = True
    file_status_indicator.object = '' # Clear previous status
    upload_btn.disabled = True # Disable during processing

    try:
        # 1. Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_name).suffix) as f:
            f.write(file_content)
            temp_path = f.name
        print(f"Temporary file created at: {temp_path}")

        # 2. Load and Split Document
        documents = load_document(temp_path) # Adds filename metadata
        if not documents:
            raise ValueError("Failed to load document or document is empty.")

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        if not texts:
            raise ValueError("Failed to split document into text chunks.")
        print(f"Split '{file_name}' into {len(texts)} text chunks.")

        # 3. Add to the main Chroma Database
        print(f"Adding {len(texts)} chunks from '{file_name}' to the vector store...")
        main_db.add_documents(texts)
        print("Chunks added successfully.")

        # 4. Persist Changes to Disk
        print("Persisting database changes...")
        main_db.persist()
        print("Database persisted.")
        num_chunks_added = len(texts)


        # 5. Update or Create the Main Retriever
        print("Updating main retriever...")
        # Use the current k_slider value when updating/creating
        main_retriever = main_db.as_retriever(search_kwargs={"k": k_slider.value})
        print(f"Main retriever updated/created. Current k={k_slider.value}. Total docs in DB: {main_db._collection.count()}")


        # 6. Update UI on Success
        file_status_indicator.object = '<span style="font-size: 1.5em; color: green;">✅</span>'

        # Prepare new row for the DataFrame
        new_file_info = {
            "File Name": [file_name],
            "Size (KB)": [round(len(file_content) / 1024, 2)],
            "Type": [Path(file_name).suffix],
            "Chunks Added": [num_chunks_added] # Add chunk info
        }
        new_row_df = pd.DataFrame(new_file_info)

        # Append to the existing DataFrame
        current_df = uploaded_files.value
        updated_df = pd.concat([current_df, new_row_df], ignore_index=True)
        uploaded_files.value = updated_df # Update the widget's value

        # Enable chat interface now that we have a retriever
        chat_input.disabled = False
        submit_btn.disabled = False
        chat_input.placeholder = "Ask questions about loaded documents..."

        print(f"Successfully processed and added '{file_name}'.")

        # Explicitly clear large objects and collect garbage
        del documents
        del texts
        gc.collect()
        print("Garbage collected.")


    except Exception as e:
        # Handle errors during processing
        file_status_indicator.object = '<span style="font-size: 1.5em; color: red;">❌</span>'
        print(f"ERROR processing {file_name}: {e}") # Log error
        # Potentially send error to chat interface if desired (but can be noisy)
        # chat_interface.send(f"Error processing {file_name}: {e}", user="System", respond=False)

    finally:
        processing_spinner.value = False # Stop spinner
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"Temporary file {temp_path} deleted.")
            except Exception as e_del:
                print(f"Error deleting temporary file {temp_path}: {e_del}")

        # Reset file input and button state
        file_input.value = None
        upload_btn.disabled = True # Disabled until a new file is selected


file_input.param.watch(update_upload_button, 'value')
upload_btn.on_click(process_uploaded_file)

def clear_chat_history(event):
    """Clear only the chat history display"""
    chat_interface.clear()
    # Send welcome message again? Or just clear? Let's just clear.
    # chat_interface.send(...)


clear_btn.on_click(clear_chat_history)

# --- Function to Clear Everything (DB and UI) ---
def clear_all_data(event):
    """Clears the persisted Chroma DB, resets the UI, and global state."""
    global main_db, main_retriever
    print("Clearing all loaded data...")

    # 1. Clear Chat Interface
    chat_interface.clear()

    # 2. Clear the Vector Database
    # Easiest way is often to delete the persistence directory and re-initialize
    try:
        print(f"Attempting to clear Chroma collection '{main_db._collection.name}'...")
        # Try clearing the collection if possible (might depend on Chroma version/API)
        # This is safer than deleting the whole directory if other things use it
        # However, Chroma's standard API doesn't always expose a simple 'clear collection'
        # A common workaround is deleting and recreating the directory.
        ids_to_delete = main_db.get()['ids']
        if ids_to_delete:
             print(f"Deleting {len(ids_to_delete)} entries from collection...")
             main_db._collection.delete(ids=ids_to_delete)
             main_db.persist() # Persist the deletion
             print("Collection entries deleted and persisted.")
        else:
             print("Collection already empty.")

        # Alternative: Delete directory (use with caution if dir is shared)
        # import shutil
        # if os.path.exists(PERSIST_DIRECTORY):
        #     shutil.rmtree(PERSIST_DIRECTORY)
        #     print(f"Deleted directory: {PERSIST_DIRECTORY}")
        # os.makedirs(PERSIST_DIRECTORY, exist_ok=True) # Recreate directory
        # Re-initialize the DB object
        # main_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        # print("Re-initialized empty Chroma DB.")

    except Exception as e:
        print(f"Error clearing Chroma DB: {e}. Manual deletion of '{PERSIST_DIRECTORY}' may be needed.")
        chat_interface.send(f"Error clearing database: {e}", user="System", respond=False)


    # 3. Reset UI State
    uploaded_files.value = initial_empty_df.copy()
    chat_input.disabled = True
    submit_btn.disabled = True
    chat_input.placeholder = "Load documents first..."
    file_status_indicator.object = ''

    # 4. Reset Global Retriever
    main_retriever = None

    # 5. Garbage Collect
    gc.collect()

    # 6. Send initial message
    chat_interface.send(
        "All loaded data has been cleared. Please select a new document.",
         user="System", respond=False
    )
    print("Data clearing process complete.")

clear_db_btn.on_click(clear_all_data)


# REFACTORED: Uses the single main_retriever
async def chat_callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    """Process user questions using the single main retriever."""
    global main_retriever # Access global retriever

    if not main_retriever:
        yield {"user": "System", "value": "Please load at least one document first using the 'Load Document' button!"}
        return

    if not contents:
        # Ignore empty input silently
        return

    processing_spinner.value = True
    submit_btn.disabled = True # Disable while processing
    chat_input.disabled = True # Disable input while processing

    try:
        # 1. Update retriever's 'k' value from slider *before* querying
        current_k = k_slider.value
        main_retriever.search_kwargs['k'] = current_k
        print(f"Using main retriever with k={current_k}")

        # 2. Initialize LLM
        try:
            llm = HuggingFaceHub(
                repo_id=LLM_MODELS[model_select.value],
                model_kwargs={"temperature": 0.1, "max_length": 512} # Increased max_length slightly
            )
        except Exception as llm_error:
            yield {"user": "System", "value": f"Error initializing LLM: {llm_error}. Is HUGGINGFACEHUB_API_TOKEN set?"}
            processing_spinner.value = False
            submit_btn.disabled = False # Re-enable on error
            chat_input.disabled = False
            return

        # 3. Create the QA chain using the Single Retriever
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_select.value,
            retriever=main_retriever, # Use the single main retriever
            return_source_documents=True
        )

        # 4. Run the query
        print(f"Invoking QA chain with query: '{contents}'")
        response = await qa.ainvoke({"query": contents})
        print("QA chain invocation complete.")

        # 5. Format and yield the response
        answer_md = f"**Answer:**\n{response['result']}"
        answer_panel = pn.Column(pn.pane.Markdown(answer_md))

        if response.get('source_documents'):
            answer_panel.append(pn.layout.Divider())
            answer_panel.append(pn.pane.Markdown("**Sources:**"))
            unique_sources = {} # Track unique sources by content snippet to avoid duplicates from overlap
            for i, doc in enumerate(response['source_documents']):
                 # Use filename and page from metadata
                 doc_filename = doc.metadata.get('filename', 'Unknown File')
                 page_num = doc.metadata.get('page', 'N/A') # Use the standardized 'page' key
                 title = f"Source Context ({doc_filename} - Page {page_num})" # Adjusted title

                 content_str = doc.page_content
                 # Only add if content is meaningfully different
                 content_key = content_str[:200] # Use first N chars as key
                 if content_key not in unique_sources:
                     unique_sources[content_key] = True

                     # Limit displayed content length in accordion preview for very long chunks
                     preview_content = (content_str[:500] + '...') if len(content_str) > 500 else content_str
                     accordion_content_md = f"```\n{preview_content}\n```"
                     accordion_content = pn.pane.Markdown(accordion_content_md)

                     answer_panel.append(
                         pn.Accordion(
                             (title, accordion_content),
                             active=False,
                             header_background="#f0f0f0",
                             margin=(5, 0)
                         )
                     )


        yield {"user": "AI Assistant", "avatar": "🤖", "value": answer_panel}

    except Exception as e:
        yield {"user": "System", "value": f"Error during question processing: {str(e)}"}
        print(f"Error in chat_callback: {e}")
    finally:
        processing_spinner.value = False
        submit_btn.disabled = False # Re-enable after processing
        chat_input.disabled = False # Re-enable input


# --- Chat Interface ---
chat_interface = pn.chat.ChatInterface(
    callback=chat_callback,
    widgets=[chat_input],
    show_rerun=False,
    show_undo=False,
    show_clear=False, # We use our custom clear buttons
    sizing_mode="stretch_width",
    height=600,
    css_classes=["chat-box"],
    message_params={"show_reaction_icons": False}
)

# Send initial welcome message
if main_db._collection.count() > 0:
     # If DB has data, reflect that
     chat_interface.send(
         f"Welcome! Found {main_db._collection.count()} existing document chunks in the database. You can ask questions or load more documents.",
         user="System", respond=False
     )
     # Enable chat input immediately if retriever is ready
     if main_retriever:
         chat_input.disabled = False
         submit_btn.disabled = False
         chat_input.placeholder = "Ask questions about loaded documents..."
         # Populate uploaded_files DataFrame (more complex - requires storing metadata)
         # For now, we just indicate data exists. A robust solution would reload the file list.
         print("INFO: Existing documents found, but UI file list is not repopulated automatically in this version.")

else:
     chat_interface.send(
         "Welcome! Please select a PDF or Word document and click 'Load Document'. You can load multiple documents.",
         user="System", respond=False
     )


# --- Layout ---
chat_controls = pn.Row(submit_btn, clear_btn, clear_db_btn, sizing_mode="stretch_width", margin=(10, 0))

file_upload_section = pn.Column(
    file_input,
    pn.Row(
        upload_btn,
        file_status_indicator,
        align='start'
    ),
    pn.pane.Markdown("**Loaded Documents Info**", margin=(10, 0, 5, 0)),
    uploaded_files, # Table now shows multiple files
    pn.layout.Divider()
)

sidebar = pn.Column(
    pn.pane.Markdown("### Configuration", styles={"font-weight": "bold"}), # Added sidebar title
    file_upload_section,
    pn.pane.Markdown("**Model & Retrieval Options**", margin=(10, 0, 5, 0)),
    model_select,
    k_slider, # Label changed to reflect overall K
    chain_select,
    processing_spinner,
    pn.layout.Divider(),
    sizing_mode="stretch_height",
    css_classes=["sidebar"]
)

main = pn.Column(
    chat_interface,
    chat_controls, # Includes clear DB button now
    sizing_mode="stretch_width",
)

template = pn.template.FastListTemplate(
    title="Multi-Document Chat Assistant (Single DB)", # Updated title
    sidebar=[sidebar],
    main=[main],
    accent_base_color="#4CAF50",
    header_background="#4CAF50",
    theme_toggle=False,
    sidebar_width=400
)

# Make the app servable
template.servable()
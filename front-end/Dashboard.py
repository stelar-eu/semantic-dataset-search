import streamlit as st
import requests

# Configure the page
st.set_page_config(
    page_title="Semantic Dataset Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🔍 Semantic Dataset Search")
st.markdown("---")

# Sidebar - Settings
with st.sidebar:
    st.header("🔧 Settings")
    
    st.subheader("Search Options")
    search_mode = st.selectbox(
        "Search Mode",
        ["Semantic", "Keyword", "Hybrid"],
        help="Choose how the search should be performed"
    )
    
    max_results = st.slider(
        "Maximum Results",
        min_value=1,
        max_value=50,
        value=10,
        help="Maximum number of results to display"
    )
    
    st.subheader("📊 About")
    st.markdown("""
    This is a semantic dataset search application that helps you find 
    relevant datasets based on natural language queries.
    
    **Features:**
    - Semantic search capabilities
    - Domain classification
    - Use case identification
    - General description extraction
    """)

# API Configuration
API_BASE_URL = "http://localhost:8000"  # Adjust this to match your server URL

def search_datasets(query, n_results=10):
    """Make API request to search_datasets_explainable endpoint"""
    try:
        url = f"{API_BASE_URL}/search_datasets_explainable"
        payload = {
            "query": query,
            "n_results": n_results,
            "auth_scope": []
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

# Search section
st.header("Search")
search_query = st.text_input(
    "Enter your search query:",
    placeholder="Search for datasets...",
    help="Describe what kind of dataset you're looking for"
)

# Search button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

st.markdown("---")

# Results section
if search_button and search_query:
    st.header("Search Results")
    
    # Show loading spinner while searching
    with st.spinner("Searching datasets..."):
        search_results = search_datasets(search_query, max_results)
    
    if search_results:
        # Display query analysis in the three columns
        query_analysis = search_results.get("query_analysis", {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📋 General Description")
            general_desc = query_analysis.get("general_description", "No description available")
            st.markdown(f"**{general_desc}**")
        
        with col2:
            st.subheader("🏷️ Domain")
            domain = query_analysis.get("domain", "No domain specified")
            st.markdown(f"**{domain}**")
        
        with col3:
            st.subheader("🎯 Use Case Purpose")
            purpose = query_analysis.get("purpose", "No purpose specified")
            st.markdown(f"**{purpose}**")
        
        st.markdown("---")
        
        # Display search results
        results = search_results.get("results", [])
        if results:
            st.subheader(f"Found {len(results)} dataset(s)")
            
            for i, (dataset_id, dataset_info) in enumerate(results, 1):
                with st.expander(f"Dataset {i}: {dataset_id}", expanded=True):
                    st.markdown("**Dataset Title:**")
                    st.write(dataset_info.get("dataset_title", "No title available"))

                    st.markdown("**Dataset Official Description:**")
                    st.write(dataset_info.get("dataset_official_description", "No official description available"))
                    
                    st.markdown("**Dataset Description:**")
                    st.write(dataset_info.get("dataset_description", "No description available"))
                    
                    st.markdown("**Use Case:**")
                    st.write(dataset_info.get("use_case", "No use case specified"))
                    
                    st.markdown("**Domain:**")
                    st.write(dataset_info.get("domain", "No domain specified"))
        else:
            st.info("No datasets found matching your query.")
    
    elif search_results is None:
        st.error("Failed to retrieve search results. Please check your connection and try again.")

elif search_button and not search_query:
    st.warning("Please enter a search query before searching.")

# Handle clear button
if clear_button:
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Semantic Dataset Search Dashboard</div>",
    unsafe_allow_html=True
)

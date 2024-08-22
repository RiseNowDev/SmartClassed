import os
import psycopg2
from psycopg2.extras import DictCursor
from typing import List, Optional
from datetime import datetime
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import StructuredTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import concurrent.futures
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the combined data structure
class CombinedData(BaseModel):
    """
    A class to represent the combined data of a supplier and item.

    Attributes:
        supplier_name (str): The name of the supplier organization.
        item_code (str): The unique code of the item (if available).
        classification_code (str): The UNSPSC classification code.
        classification_name (str): The UNSPSC classification name.
        website (str): The website of the supplier or related to the item.
        comments (str): Any additional comments.
        valid (bool): Whether the combination is valid.
    """

    supplier_name: str = Field(description="The name of the supplier organization")
    item_code: Optional[str] = Field(description="The item description that is sold by the supplier")
    classification_code: str = Field(description="The UNSPSC classification code")
    classification_name: str = Field(description="The UNSPSC classification name")
    website: str = Field(description="The website of the supplier or related to the item")
    comments: str = Field(description="Any additional comments")
    valid: bool = Field(description="Whether the combination is valid")

# Initialize the ChatOpenAI client
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Initialize the GoogleSerperAPIWrapper tool
google_search = GoogleSerperAPIWrapper(
    api_key=os.getenv("SERPER_API_KEY"),
    gl="us",
    hl="en",
    type="search",
)

# Create the parser
parser = PydanticOutputParser(pydantic_object=CombinedData)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI assistant tasked with gathering and accurately classifying information about supplier companies and their items.",
    ),
    ("human", 
     "I need detailed information about the supplier: {supplier_name} and the item code: {item_code}. "
     "Please ensure that the classification is as accurate as possible."),
    (
        "system",
        "Certainly! I'll use available tools and references, including similar industry items and suppliers, "
        "to search for information about the supplier {supplier_name} and the item with code {item_code}. "
        "Please provide only factual information that you can verify. If you cannot find exact information, "
        "use the closest possible classification based on similar items. Do not generate random guesses. "
        "Provide the following details:\n"
        "1. The UNSPSC classification code and name (inferred if not directly available)\n"
        "2. The website of the supplier or related to the item\n"
        "3. Any additional relevant comments\n"
        "4. Validation of whether it's a valid supplier-item combination\n\n"
        "Format the information as follows:\n"
        "{format_instructions}",
    ),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Global variables for statistics
server_count = 0
openai_count = 0
total_affected = 0

def process_supplier_item(supplier_name: str, item_code: Optional[str]) -> CombinedData:
    """
    Process the supplier name and item code to return the combined data.

    Args:
        supplier_name (str): The name of the supplier.
        item_code (Optional[str]): The code of the item (if available).

    Returns:
        CombinedData: The combined supplier and item data.
    """
    global server_count, openai_count

    tools = [
        StructuredTool.from_function(
            name="investigate_supplier_item",
            func=google_search.run,
            description="Use Google search to find information about the supplier and item.",
        )
    ]

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    server_count += 1  # Increment server count
    openai_count += 1  # Increment OpenAI count

    result = agent_executor.invoke(
        {
            "supplier_name": supplier_name,
            "item_code": item_code if item_code else "N/A",
            "format_instructions": parser.get_format_instructions(),
        }
    )
    parsed_data = parser.parse(result["output"])
    logging.info(f"Processed supplier {supplier_name} and item {item_code}: {parsed_data}")
    return parsed_data

def get_items_to_process(cursor, batch_size: int) -> List[tuple]:
    """
    Retrieve items that need processing from the database.

    Args:
        cursor: The database cursor.
        batch_size (int): The number of items to retrieve.

    Returns:
        List[tuple]: A list of tuples containing (id, supplier_name, item_code) for items to process.
    """
    query = """
    SELECT id, supplier_name, item_code
    FROM supplier_items
    WHERE classification_code IS NULL AND lookup IS NULL AND spend::numeric > 1000
    LIMIT %s
    """
    cursor.execute(query, (batch_size,))
    return cursor.fetchall()

def update_item_info(conn, item_id: int, combined_data: CombinedData):
    """
    Update item information in the database.

    Args:
        conn: The database connection.
        item_id (int): The ID of the item to update.
        combined_data (CombinedData): An instance of CombinedData containing the updated data.
    """
    global total_affected
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            UPDATE supplier_items
            SET classification_code = %s, classification_name = %s, 
                website = %s, comments = %s, valid = %s
            WHERE id = %s
            """,
            (
                combined_data.classification_code,
                combined_data.classification_name,
                combined_data.website,
                combined_data.comments,
                combined_data.valid,
                item_id,
            ),
        )
        affected_rows = cursor.rowcount
        conn.commit()
        total_affected += affected_rows
        logging.info(f"Updated item {item_id}. Affected rows: {affected_rows}")
    except Exception as e:
        logging.error(f"Error updating item {item_id}: {str(e)}")
        conn.rollback()

def process_single_item(id: int, supplier_name: str, item_code: Optional[str], conn):
    """
    Process a single item by retrieving and updating its information.

    Args:
        id (int): The ID of the item.
        supplier_name (str): The name of the supplier.
        item_code (Optional[str]): The code of the item (if available).
        conn: The database connection.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    try:
        logging.info(f"Processing supplier: {supplier_name}, item: {item_code}")
        combined_data = process_supplier_item(supplier_name, item_code)
        update_item_info(conn, id, combined_data)
        return True
    except Exception as e:
        logging.error(f"Error processing supplier {supplier_name}, item {item_code}: {str(e)}")
        return False

def insert_run_stats(conn, run_start: datetime, run_end: datetime):
    """
    Insert run statistics into the run_stats table.

    Args:
        conn: The database connection.
        run_start (datetime): The start time of the run.
        run_end (datetime): The end time of the run.
    """
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO run_stats (run_start, run_end, serper_count, openai_count, total_affected)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (run_start, run_end, server_count, openai_count, total_affected)
        )
        conn.commit()
        logging.info("Run statistics inserted successfully")
    except Exception as e:
        logging.error(f"Error inserting run statistics: {str(e)}")
        conn.rollback()

def process_items(batch_size: int = 100):
    """
    Main function to process items in batches.

    Args:
        batch_size (int): The number of items to process in one batch. Default is 100.
    """
    global server_count, openai_count, total_affected
    
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="ars_spend_class",
        user="overlord",
        password="password"
    )
    conn.set_session(autocommit=False)
    cursor = conn.cursor(cursor_factory=DictCursor)

    run_start = datetime.now()

    try:
        # Retrieve items that need processing
        items = get_items_to_process(cursor, batch_size)

        # Use a thread pool to process items concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(process_single_item, id, supplier_name, item_code, conn)
                for id, supplier_name, item_code in items
            ]

            # Count the number of successfully processed items
            successful = sum(future.result() for future in concurrent.futures.as_completed(futures))

        logging.info(f"Successfully processed {successful} out of {len(items)} items")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        run_end = datetime.now()
        insert_run_stats(conn, run_start, run_end)
        cursor.close()
        conn.close()

    logging.info(f"Run completed. Server count: {server_count}, OpenAI count: {openai_count}, Total affected: {total_affected}")

# Example usage
if __name__ == "__main__":
    process_items(batch_size=100000)  # Process 100 items at a time
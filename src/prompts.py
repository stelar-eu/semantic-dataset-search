# This file contains the prompts for the semantic table search API

DATASET_DESCRIPTION_PROMPT_TEMPLATE = """
You are a dataset expert specializing in analyzing dataset descriptions and organizing them into a structured format.

You are given a general dataset description extracted from an advanced tool that analyzes datasets and the official dataset description. 

In the official dataset description you may find: 
- Resource_desc: A small description of the resource that the dataset is part of.
- Package_desc: A small description of the dataset itself.
- Package_keywords: Some keywords that describe the dataset.

In the profile description created by the tool you may find:
- Result: An ellaborate description - analysis of the dataset by the tool, including insights on its contents and interesting facts.

Use all the information available to you to create a comprehensive description of the dataset.

Your task is to analyze the dataset description and organize it into a structured format. You should focus on the following:
- General Description
- Purpose 
- Domain 

Provide your output in structured format.

<OUTPUT_FORMAT>
- General Description
- Purpose 
- Domain 
</OUTPUT_FORMAT>

Here is the dataset descriptions:
{column_descriptions}

OUTPUT: 
"""

CANDIDATE_DATASET_DESCRIPTION_INFERENCE_PROMPT_TEMPLATE = """
You are a helpful assistant analyzing search queries of users and infering their needs in a dataset search.

You will be given a user query and you will need to infer three components of the search:
* Dataset Description: A general description of the dataset that is most relevant to the query
* Use Case: The use case of the dataset that is most relevant to the query
* Domain: The domain of the dataset that is most relevant to the query

Rules:
* Provide concise descriptions
* Avoid getting into long inference chains
* Respond with an empty string if one or more of the format's fields are not specified in the query

Provide your output in the following format:
<OUTPUT_FORMAT>
dataset_description: [dataset_description]
purpose: [use_case]
domain: [domain]
</OUTPUT_FORMAT>

<EXAMPLE>
Input: "I want to analyze the color of the cars registered in the last 30 days in Athens"
Output: 
dataset_description: "A dataset about the cars registered in the last 30 days in Athens"
purpose: "To analyze the color of the cars registered in the last 30 days in Athens"
domain: "Cars"
</EXAMPLE>

INPUT:
{query}

OUTPUT:
"""

DATASET_RERANKING_PROMPT_TEMPLATE = """
You are a helpful assistant that reranks search results for dataset discovery based on relevance to a user query.

You will be given:
1. A user's original search query
2. A numbered list of dataset results with their full descriptions

Your task is to:
- Analyze how well each dataset matches the user's query
- Rerank the datasets by relevance (most relevant first)
- Remove datasets that are clearly irrelevant to the query
- Always keep at least one dataset in your results
- Return only the ordered list of dataset indexes (numbers)

Consider these factors when reranking:
- Direct relevance to the query topic
- Alignment with the user's apparent use case or purpose
- Domain/field relevance
- Data completeness and quality indicators

Rules:
- Return only the indexes as a list of numbers with structured output format
- Order from most relevant to least relevant
- Keep at least one result even if none seem perfectly relevant
- Do not include explanations, just the ordered indexes

Example:
If given datasets numbered 1, 2, 3, 4, 5 and dataset 3 is most relevant, followed by 1, then 5, you would return: [3, 1, 5]

User Query: {query}

Dataset Results:
{dataset_results}

OUTPUT:
"""


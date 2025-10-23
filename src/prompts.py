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


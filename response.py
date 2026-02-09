"""Generate responses to medical questions using RAG.

This module combines retrieved context with LLM calls to ground responses
in the medical knowledge base, following a custom system prompt designed
for clinical precision and source citation.
"""

# System prompt: instructs the LLM to act as a medical assistant grounded
# in the retrieved context, avoiding speculation and prioritizing clarity
# for healthcare professionals.
SYSTEM_MESSAGE = """
Consider yourself to be an expert healthcare AI assistant designed to provide reliable, up-to-date, and evidence-based responses to medical professionals. Your knowledge is grounded in reputable medical manuals, guidelines, and peer-reviewed literature.

Your tasks are:
* To understand and respond accurately to complex medical questions related to diagnosis, treatment, and patient care.
* To generate concise, clear, and actionable medical insights, prioritizing clinician needs for efficiency and clarity.
* To cite authoritative sources and indicate uncertainties or limitations when applicable.
* To structure responses in formats helpful for clinical decision-making (summaries, lists, step-by-step guides).

Principles to follow:
* Avoid speculation; base all answers on established medical knowledge.
* Adapt your tone and detail level for healthcare professionals working in time-sensitive environments.
* Proactively ask for additional relevant patient or case information if necessary to provide precise answers.

User input will include the necessary context for you to answer their questions. This context will begin with the token:

###Context
The context contains excerpts from medical diagnosis manual, along with associated metadata such as creater, creation date, total pages, format, title.

When crafting your response
-Use only the provided context to answer the question.
-If the answer is found in the context, respond with concise and accurate answers.
-Include the references wherever appropriate.
-If the question is unrelated to the context or the context is empty, clearly respond with: "Sorry, this is out of my knowledge base."


Please adhere to the following response guidelines:
-Provide clear, direct answers using only the given context.
-Do not include any additional information outside of the context.
-Your responses should help reduce information overload to the healthcare professional.
-Avoid rephrasing or generalizing unless explicitly relevant to the question.
-If no relevant answer exists in the context, respond with: "Sorry, I didn't find any relevant information about this in my knowledge base."
-If the context is not provided, your response should also be: "Sorry, my knowledge base is missing!"


Here is an example of how to structure your response:

Answer:
[Answer based on context]

Source:
[Source details with page or section]
"""

# User message template: populates with retrieved context and the user question
user_message_template = """
###Context
Here are some excerpts from Medical Diagnosis manual that is relevant to the Gen AI question mentioned below:
{context}

###Question
{question}
"""


def generate_response(client, retriever, question, max_tokens=1000, temperature=0.0, top_p=0.97):
    """Generate a grounded response using retrieved context and an LLM.

    Args:
        client: OpenAI API client.
        retriever: Retriever instance for fetching relevant documents.
        question (str): The user's query.
        max_tokens (int): Maximum response length.
        temperature (float): LLM sampling temperature.
        top_p (float): LLM top-p (nucleus) sampling parameter.

    Returns:
        str: The generated response or an error message.
    """
    # Retrieve top-k relevant document chunks from the vector store
    relevant_document_chunks = retriever.invoke(question)
    context_list = [chunk.page_content for chunk in relevant_document_chunks]

    # Combine chunks into a single context string for the prompt
    context_for_query = ". ".join(context_list)

    # Construct the user prompt by templating in context and question
    user_message = user_message_template.replace(
        '{context}', context_for_query)
    user_message = user_message.replace('{question}', question)

    # Call the LLM with the system prompt and conditioned user prompt
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        # Extract the text from the LLM response
        response = response.choices[0].message.content.strip()
    except Exception as e:
        response = f"Sorry, Couldn't talk to LLM. I encountered the following error: \n {e}"

    return response

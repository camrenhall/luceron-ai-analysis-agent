# Workflow Execution Prompt Template

Execute document analysis workflow:

Workflow ID: {workflow_id}
Case ID: {case_id}        
Documents: {document_ids}
Case Context: {case_context}

Start by creating an analysis plan, then execute analysis using OpenAI o3. Use the case ID "{case_id}" when retrieving case context. When analyzing documents, always include the workflow_id "{workflow_id}" in your analysis requests.
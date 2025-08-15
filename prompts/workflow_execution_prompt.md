# Workflow Execution Prompt Template

Execute document analysis workflow:

Case ID: {case_id}        
Documents: {document_ids}
Case Context: {case_context}

Start by creating an analysis plan, then execute analysis using OpenAI o3. Use the case ID "{case_id}" when retrieving case context.
# Document Satisfaction Criteria

This document defines the rules and criteria for determining when a submitted document satisfies a requested document requirement. The Analysis Agent uses these criteria to automatically mark documents as "completed" when they meet the specified requirements.

## General Satisfaction Rules

### 1. Document Type Matching
- The submitted document type must match the requested document type
- Partial matches or similar document types generally do not satisfy requirements
- Example: A "W2" request is satisfied by a "W2" document, not by a "1099" or "Paystub"

### 2. Date/Year Specificity
- If a specific year is requested, the submitted document must be from that exact year
- Example: "2016 Federal Tax Return" is only satisfied by a 2016 Federal Tax Return, not 2017 or 2018
- Multi-year requests: "2020-2022 Tax Returns" requires all three years to be considered complete

### 3. Completeness Requirements
- Single-page documents from multi-page sets do not satisfy complete document requests
- Example: "Complete Bank Statement" requires all pages, not just the first page

## Specific Document Types

### Tax Documents
#### Federal Tax Returns
- **Request**: "Federal Tax Return [YEAR]"
- **Satisfies**: Complete Form 1040 (and schedules if applicable) for the specified year
- **Does Not Satisfy**: 
  - Different year
  - State tax returns only
  - Partial returns (missing schedules when required)
  - Tax transcripts (unless specifically requested)

#### State Tax Returns  
- **Request**: "State Tax Return [YEAR]"
- **Satisfies**: Complete state tax return for the specified year and state
- **Does Not Satisfy**:
  - Federal returns only
  - Different year
  - Different state (if state is specified)

#### W-2 Forms
- **Request**: "W2" or "W-2 [YEAR]"
- **Satisfies**: Official W-2 form from employer for specified year
- **Does Not Satisfy**: 
  - 1099 forms
  - Paystubs
  - Different year

#### 1099 Forms
- **Request**: "1099" or "1099-[TYPE]"
- **Satisfies**: Specific 1099 form type (1099-MISC, 1099-INT, etc.)
- **Does Not Satisfy**:
  - Different 1099 type (unless general "1099" requested)
  - W-2 forms
  - Bank statements

### Financial Statements
#### Bank Statements
- **Request**: "Bank Statement [PERIOD]"
- **Satisfies**: Complete bank statement for specified period
- **Does Not Satisfy**:
  - Different period/month
  - Partial statements
  - Check images only
  - Account summaries without transactions

#### Credit Card Statements
- **Request**: "Credit Card Statement [PERIOD]"
- **Satisfies**: Complete credit card statement for specified period
- **Does Not Satisfy**:
  - Bank statements
  - Different period
  - Partial statements

#### Investment/Brokerage Statements
- **Request**: "Investment Statement" or "Brokerage Statement [PERIOD]"
- **Satisfies**: Complete investment account statement for specified period
- **Does Not Satisfy**:
  - Bank statements
  - Different period
  - Trade confirmations only

### Employment Documents
#### Paystubs
- **Request**: "Paystub [PERIOD]" or "Recent Paystubs"
- **Satisfies**: 
  - Specific period: Paystub from exact period requested
  - Recent: Paystubs from last 30-90 days
- **Does Not Satisfy**:
  - W-2 forms
  - Employment verification letters
  - Different period

#### Employment Verification
- **Request**: "Employment Verification" or "Letter of Employment"
- **Satisfies**: Official letter from employer confirming employment
- **Does Not Satisfy**:
  - Paystubs only
  - W-2 forms
  - Offer letters

### Property Documents
#### Deeds
- **Request**: "Property Deed" or "Deed"
- **Satisfies**: Official recorded property deed
- **Does Not Satisfy**:
  - Purchase agreements
  - Property tax statements
  - Appraisals

#### Mortgage Statements
- **Request**: "Mortgage Statement [PERIOD]"
- **Satisfies**: Official mortgage statement for specified period
- **Does Not Satisfy**:
  - Loan documents
  - Property tax statements
  - Different period

### Insurance Documents
#### Insurance Policies
- **Request**: "Insurance Policy [TYPE]"
- **Satisfies**: Complete insurance policy document for specified type
- **Does Not Satisfy**:
  - Insurance cards/declarations only
  - Different insurance type
  - Expired policies (unless historical data requested)

## Special Cases and Exceptions

### 1. Equivalent Documents
Some documents may be considered equivalent for certain purposes:
- **Tax Transcripts**: May satisfy "Tax Return" requests if specifically noted as acceptable
- **Bank Statements vs. Deposit Slips**: Generally not equivalent unless specifically noted

### 2. Multiple Document Requests
When multiple documents are requested as a group:
- **"Last 3 Bank Statements"**: Requires all 3 consecutive statements
- **"2020-2022 Tax Returns"**: Requires returns for all 3 years
- **"W-2s from All Employers"**: Requires W-2s from each employer mentioned/known

### 3. Range Requests
For time-based ranges:
- **"Bank Statements Jan-Dec 2023"**: Requires statements covering the entire period
- **"Recent Paystubs"**: Typically 2-3 most recent paystubs (last 60-90 days)

## Partial Satisfaction Handling

### Documents Not Yet Complete
- Mark as "partially satisfied" if some but not all requirements are met
- Example: 2 of 3 requested bank statements submitted
- Continue tracking remaining requirements

### Quality Issues
Documents may be submitted but not satisfy requirements due to quality:
- **Illegible**: Document cannot be read clearly
- **Incomplete**: Missing pages or critical information
- **Unofficial**: Copies that appear altered or non-official

## Configuration Notes

### Customizable Criteria
This document can be modified to adjust satisfaction criteria for specific cases or client requirements:

1. **Time Windows**: Adjust how recent "recent documents" need to be
2. **Equivalency Rules**: Add or remove document types that can satisfy certain requests
3. **Quality Standards**: Modify what constitutes acceptable document quality
4. **Specificity Requirements**: Adjust how specific date/period matching needs to be

### Implementation Guidelines
- The Analysis Agent should reference this document when evaluating document satisfaction
- Criteria should be applied consistently across all cases
- When in doubt, err on the side of requiring manual review rather than auto-completing
- Log all satisfaction decisions for audit and improvement purposes

## Version History
- **v1.0**: Initial criteria document (2024)
- Last Updated: [Current Date]
- Next Review: [Quarterly Review Date]
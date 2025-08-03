# UK Fraud Call Transcript Analysis

## Objective
Generate synthetic call transcripts mimicking real UK fraud cases (e.g., account takeover, identity theft, authorised push payment, vishing, fraudster disguised as customer) 
using Gemini 2.5 Flash (free tier). Analyze these for fraud modus operandi, customer sentiment, and new feature recommendations to improve fraud detection models.

## Use Case
- Identify fraud patterns to reduce false positives in detection models.
- Propose new features for existing fraud detection models.
- Develop tailored customer notifications or dashboards for fraud awareness.

## Tools
- **Python**: For scripting and data processing.
- **Gemini 2.5 Flash**: Free tier (~60 requests/minute, 1M token context window) for transcript generation and analysis.
- **JupyterLab**: For interactive development and analysis.
- **GitHub**: For version control and collaboration.

## Constraints
- Free and scalable solution.
- Designed to run on a MacBook Air M4.

## Future Scope
- Utilize real transcript data to derive fraud modus operandi.
- Create new features for fraud detection models.
- Design tailored communication nudges (e.g., SMS alerts, email warnings).

## Project Phases

### Phase 1: Generate Synthetic Call Transcripts
- **Objective**: Generate realistic transcripts for UK-specific fraud scenarios (account takeover, identity theft, authorised push payment, vishing, fraudster disguised as customer).
- **Tasks**:
  - Include fraud indicators (e.g., urgency, vague responses) and UK banking terms (e.g., Faster Payments, sort code).
  - Incorporate agent investigative questions (e.g., verification, transaction details).
  - Vary transcript durations:
    - Short: ~100 words
    - Medium: ~200 words
    - Long: ~300 words
  - Save transcripts as text files.
  - Store metadata (Transcript_ID, Scenario, Duration, Word_Count, Fraud_Indicators, File_Path) in a CSV for Phase 2 analysis.

### Phase 2: Analyze Transcripts for Fraud Modus Operandi and Feature Recommendations
- **Objective**: Identify fraud patterns and recommend new features.
- **Tasks**:
  - Analyze transcripts for patterns (e.g., repeated keywords, evasive behavior).
  - Use Gemini 2.5 Flash to propose new features based on patterns and raw variables (to be provided).
  - Save analysis results for Phase 3 model building.

### Phase 3: Build and Evaluate Fraud Detection Model
- **Objective**: Develop and evaluate a fraud detection model with new features.
- **Tasks**:
  - Incorporate existing and new features into the model.
  - Compare performance against the current model to assess improvements.
  - Design tailored customer nudges (e.g., fraud alerts) based on findings.

# Phase 1: Objective and Execution
### Objective: 
Generate a realistic UK bank call transcript for Account Takeover (ATO) to test and refine prompt design, then replicate for other fraud types: Card-not-present (CNP) Fraud, Card-present Fraud, Application Fraud, Social Engineering & Impersonation, Authorised Push Payment (APP) Fraud, and Other Notable Forms. Later, generate 3-4 transcripts per fraud type with varied durations.
### Details:
- Include fraud indicators (e.g., urgency, vague responses), UK banking terms (e.g., Faster Payments, sort code), and agent investigative questions (e.g., verification, transaction details).
- Vary durations: short (~3-5 min, ~150-250 words), medium (~7-10 min, ~350-500 words), long (~10+ min, ~500+ words).
- Generate a dynamic fraud modus operandi (MO) for ATO using a separate prompt, include it in the transcript prompt, and save it in the CSV.
- Save transcript as a text file and metadata (Transcript_ID, Scenario, Duration, Word_Count, Fraud_Indicators, File_Path, Fraud_Modus_Operandi) in a CSV.
- Use a generic UK bank.
### Phase 1 Scope:
Start with one ATO transcript with a dynamically generated MO, iterate for realism, then expand to 3-4 transcripts per fraud type.

### Execution Steps
- Setup Environment
- Navigate to project folder, activate virtual environment, install google-generativeai and pandas.
- Set Gemini API key in ~/.zshrc.
- Design Prompts:
  Create a prompt to generate a realistic ATO fraud modus operandi.
  Craft a transcript prompt for ATO, embedding the generated MO, ensuring UK context, realistic dialogue, and investigative questions.
- Generate and Iterate: Use Gemini 2.5 Flash to generate one ATO transcript with the dynamic MO, iterate based on realism feedback.
- Save Outputs: Save transcript in outputs/transcripts/ as .txt.
  Save metadata, including dynamic MO, in outputs/transcripts_metadata.csv.
- Sync with GitHub:
  Commit notebook and outputs to GitHub.

# Phase 2: Using Transcript to Generate features
### Objective: 
Use Gemini 2.5 Flash to analyze synthetic Account Takeover (ATO) transcripts from outputs/transcripts_metadata.csv and text files to generate and validate fraud modus operandi, then recommend advanced features for the existing fraud detection model to address missed frauds (approved transactions reported by customers via inbound calls), reducing false positives.
### Inputs:
transcripts_metadata.csv and transcript text files (Transcript_ID, Scenario, Duration, Word_Count, File_Path, Fraud_Indicators, Fraud_Modus_Operandi).
Existing fraud model features and raw variables (to be provided).
### Key Tasks:
- Extract Fraud Patterns and Modus Operandi: Analyze transcripts for patterns (e.g., urgency, vague responses, keywords like "suspicious"), generate a 1-2 line modus operandi, and validate it against the provided Fraud_Modus_Operandi in the CSV.
- Generate Advanced Feature Recommendations: Use Gemini 2.5 Flash to recommend sophisticated features (beyond average intelligence) based on fraud patterns, generated MO, and provided raw variables, explaining why the fraud was missed. Include feature name, description, required raw variables, and a BigQuery SQL script to create each feature.
- Save Results: Store analysis (patterns, generated MO, validation result) and feature recommendations in a CSV for Phase 3.
### UK Context:
Align with UK banking terms (e.g., Faster Payments, sort code) and ATO fraud scenarios.
### Constraints: 
Use free tools (Gemini 2.5 Flash, Python, JupyterLab), scalable, runs on MacBook Air M4.
### Future Scope: 
Feed results into Phase 3 for model development and tailored customer nudges.
### Assumptions:
- Transcripts represent inbound calls reporting missed frauds.
- Existing model scores transactions, applies rules, and misses some frauds.
- Feature recommendations should be innovative, leveraging your provided model features and raw variables.

# Sequential Steps for Phase 2 Development
To ensure a structured and manageable approach to Phase 2, I’ve broken it down into small, sequential steps. This keeps the process clear and focused, aligning with the requirement for concise, step-by-step solutions.

### Provide Existing Model Features and Raw Variables:
### Extract Fraud Patterns and Generate Modus Operandi:
- Analyze transcripts from outputs/transcripts_metadata.csv and text files to extract fraud patterns (e.g., keywords, urgency).
- Generate a 1-2 line modus operandi for each transcript and validate against the provided Fraud_Modus_Operandi.
- Output: Save patterns and generated MO in an intermediate CSV.
### Recommend Advanced Features:
- Use Gemini 2.5 Flash to recommend sophisticated features based on patterns, generated MO, and provided raw variables, explaining why the fraud was missed.
- Include feature name, description, required raw variables, and a BigQuery SQL script.
- Output: Save recommendations in a final CSV for Phase 3.
### Validate and Save Results:
- Combine analysis (patterns, MO, validation results) and feature recommendations into a single CSV.
- Sync with GitHub (https://github.com/svdp2304/genai_transcript).

# Fraud Type: CNP
## Step 1: Generate 5 Card Not Present (CNP) Fraud Modus Operandi
### Objective: 
Generate 5 unique CNP fraud modus operandi using Gemini 2.5 Flash, each describing a distinct method of perpetrating CNP fraud (e.g., stolen card details, online phishing, skimming). Save the results in a CSV for use in subsequent steps.

### Approach: 
Use Gemini 2.5 Flash to generate the modus operandi, similar to the ATO process, ensuring alignment with UK banking context (e.g., Faster Payments, sort code). Store results with columns: Case_ID, Scenario, Fraud_Modus_Operandi.


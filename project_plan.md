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

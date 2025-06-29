## USAGE

uv run doc-eval evaluate examples/sample_document.txt examples/comprehensive_criteria.json

## RESULT

───────────────── 🚀 DocEval - Document Evaluation System ──────────────────
✅ Connected to DocEval MCP Server
⠸ 🤖 Running AI evaluation...INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/chat "HTTP/1.1 200 OK"
WARNING:doc_eval.server:Could not parse structured response: No JSON found in response
⠼ ✨ Finalizing results...
────────────────────── 📊 Document Evaluation Results ──────────────────────
╭────────────────────────── Document Information ──────────────────────────╮
│ 📄 **Document:** examples/sample_document.txt                            │
│ 📝 **Format:** TXT                                                       │
│ 📊 **Words:** 626                                                        │
│ 🔤 **Characters:** 4,896                                                 │
╰──────────────────────────────────────────────────────────────────────────╯
╭────────────────────────── Evaluation Criteria ───────────────────────────╮
│ 📋 **Criteria File:** examples/comprehensive_criteria.json               │
│ 🎯 **Criteria Count:** 10                                                │
│ ⚖️ **Total Weight:** 18.5                                                 │
╰──────────────────────────────────────────────────────────────────────────╯
                            📈 Individual Scores                            
┏━━━━━━━━━━━━━━━━━━━━━━┳━━┳━━┳━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Criterion            ┃  ┃  ┃  ┃ Justification                            ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━╇━━╇━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Technical Accuracy   │  │  │  │ Could not parse detailed evaluation for  │
│                      │  │  │  │ Technical Accuracy                       │
│ Clarity and          │  │  │  │ Could not parse detailed evaluation for  │
│ Readability          │  │  │  │ Clarity and Readabil...                  │
│ Completeness         │  │  │  │ Could not parse detailed evaluation for  │
│                      │  │  │  │ Completeness                             │
│ Organization and     │  │  │  │ Could not parse detailed evaluation for  │
│ Structure            │  │  │  │ Organization and Str...                  │
│ Innovation and       │  │  │  │ Could not parse detailed evaluation for  │
│ Insight              │  │  │  │ Innovation and Insig...                  │
│ Evidence and Support │  │  │  │ Could not parse detailed evaluation for  │
│                      │  │  │  │ Evidence and Support                     │
│ Audience             │  │  │  │ Could not parse detailed evaluation for  │
│ Appropriateness      │  │  │  │ Audience Appropriate...                  │
│ Visual Design and    │  │  │  │ Could not parse detailed evaluation for  │
│ Formatting           │  │  │  │ Visual Design and Fo...                  │
│ Conciseness          │  │  │  │ Could not parse detailed evaluation for  │
│                      │  │  │  │ Conciseness                              │
│ Actionability        │  │  │  │ Could not parse detailed evaluation for  │
│                      │  │  │  │ Actionability                            │
└──────────────────────┴──┴──┴──┴──────────────────────────────────────────┘
╭─────────────────────────── Overall Assessment ───────────────────────────╮
│ 🏆 **Overall Score:** 5.0/10                                             │
│ ⚖️ **Weighted Score:** 5.0                                                │
│ 📊 **Percentage:** 50.0%                                                 │
│                                                                          │
│ 📝 **Summary:**                                                          │
│ Evaluation completed but response parsing failed                         │
╰──────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────── 💡 Recommendations ───────────────────────────╮
│ • Retry evaluation with different model                                  │
╰──────────────────────────────────────────────────────────────────────────╯
╭────────────────────────── 📊 Advanced Metrics ───────────────────────────╮
│ 📊 **Mean Score:** 5.00                                                  │
│ 📈 **Std Dev:** 0.00                                                     │
│ 🎯 **Consistency:** 1.00                                                 │
│ 📉 **Range:** 0.0                                                        │
│                                                                          │
│ 🌟 **Excellent:** 0                                                      │
│ ✅ **Good:** 0                                                           │
│ ⚠️ **Fair:** 0                                                            │
│ ❌ **Poor:** 10                                                          │
│                                                                          │
│ 🚨 **Risk Level:** Medium                                                │
│ ⚠️ **Critical Issues:** 0                                                 │
│ 🔍 **Concerning Areas:** 10                                              │
╰──────────────────────────────────────────────────────────────────────────╯

✅ Evaluation completed successfully!


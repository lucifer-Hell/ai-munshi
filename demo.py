import gradio as gr
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, re

# === Load model
model_path = "./munshi-ai"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True
)



# === When CSV is uploaded
def preview_csv(file):
    try:
        raw_df = pd.read_csv(file.name)
        return raw_df.head()
    except Exception as e:
        return pd.DataFrame([["❌ Error loading file:", str(e)]])


# === Helper: Generate code from query
def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=150, temperature=0.7, top_p=0.9, do_sample=True
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            return result.get("df_code", ""), result
        except:
            return "", {"error": "Could not parse JSON"}
    return "", {"error": "No JSON found"}

# === Main logic
def handle_query(file, query):
    df = pd.read_csv(file.name)
    columns= df.columns
    prompt = (
        f"<|system|>\nYou are an expert Python assistant. Generate valid Pandas code based on the user's query.\n"
        f"The DataFrame contains the following columns: {columns}\n<|end|>\n"
        f"<|user|>\n{query}\n<|end|>\n<|assistant|>\n"
    )
    df_code, raw = generate_code(prompt)

    print("df code ",df_code)

    if not df_code:
        return "❌ Could not generate code.", pd.DataFrame(), str(raw)

    try:
        local_env = {"df": df.copy()}
        exec(f"result = {df_code}", {}, local_env)
        result_df = local_env.get("result", None)

        # Fallback to df if result is missing or not a DataFrame/Series
        if not isinstance(result_df, (pd.DataFrame, pd.Series)):
            result_df = local_env.get("df", pd.DataFrame())

        # If it's a Series, convert to DataFrame
        if isinstance(result_df, pd.Series):
            result_df = result_df.to_frame()

        return f"✅ Code: {df_code}", result_df, str(raw)
    except Exception as e:
        return f"⚠️ Code Error: {e}", pd.DataFrame(), str(raw)


# === Interface
with gr.Blocks(title="AI-Powered Pandas Assistant") as demo:
    gr.Markdown("## 🧠 Upload a CSV and Ask Pandas Questions")

    with gr.Row():
        file_input = gr.File(label="Upload CSV", file_types=[".csv"])
        sheet_preview = gr.Dataframe(label="📊 Sheet Preview")

    file_input.change(fn=preview_csv, inputs=file_input, outputs=sheet_preview)

    query = gr.Textbox(label="Ask a query about the data")
    submit_btn = gr.Button("Generate & Execute")

    code_output = gr.Textbox(label="Generated Pandas Code")
    result_table = gr.Dataframe(label="Result of Execution")
    model_json = gr.Textbox(label="Model Output", lines=3)

    submit_btn.click(
        fn=handle_query,
        inputs=[file_input, query],
        outputs=[code_output, result_table, model_json]
    )

demo.launch()
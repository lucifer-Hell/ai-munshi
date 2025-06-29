import gradio as gr
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
import torch, json, re

# === Load model
model_path = "./model/codet5p-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(
    model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True
).to("cpu")



# === When CSV is uploaded
def preview_csv(file):
    try:
        raw_df = pd.read_csv(file.name)
        return raw_df.head()
    except Exception as e:
        return pd.DataFrame([["❌ Error loading file:", str(e)]])


# === CodeT5-friendly code generation ===
def generate_code(query, max_new_tokens=128):
    input_ids = tokenizer(query, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # deterministic
        num_beams=5,  # better quality (optional)
        early_stopping=True
    )
    generated_code = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_code, {"raw_output": generated_code}

# === Main query handler for CodeT5 ===
def handle_query(file, query: str):
    df = pd.read_csv(file.name, on_bad_lines="skip")
    columns = list(df.columns)

    # Construct prompt in CodeT5's expected "source" format
    prompt = f"The DataFrame contains the following columns: {', '.join(columns)}.\nQuery: {query}"

    df_code, raw = generate_code(prompt)
    print("Generated Code:", df_code)

    if not df_code:
        return "❌ Could not generate code.", pd.DataFrame(), str(raw)

    try:
        local_env = {"df": df.copy()}
        exec(f"result = {df_code}", {}, local_env)
        result_df = local_env.get("result", None)

        if not isinstance(result_df, (pd.DataFrame, pd.Series)):
            result_df = local_env.get("df", pd.DataFrame())

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
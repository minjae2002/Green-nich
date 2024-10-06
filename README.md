# 그리니치 Green-Niche

이 서비스는 사용자와의 자연스러운 대화를 통해 ESG(환경, 사회, 지배구조) 기준에 대한 명확한 답변을 제공하여 정보 검색의 효율성을 높이는 것을 목표로 합니다. 특히, ESRS(유럽 지속 가능성 보고 표준)와 GRI(글로벌 보고 이니셔티브) 각각의 기준 및 그 사이의 변환에 대한 전문적인 지식을 제공하여 사용자의 이해를 돕습니다.

## 주요 기능
- **질문 응답 시스템**: BERT 및 RAG 모델을 활용하여 문서에서 관련 정보를 검색하고, 사용자 질문에 대해 정확하고 신속한 답변을 생성합니다. PEFT와 QLoRA 기술을 통합하여 모델의 성능을 향상시킵니다.
- **API 및 웹 통합**: OpenAI API를 사용하여 대화형 응답을 생성하고, Streamlit 기반의 사용자 친화적인 웹 인터페이스를 통해 쉽게 질문을 입력하고 답변을 받을 수 있도록 합니다.
- **문서 및 대화 관리**: 다양한 형식의 문서를 업로드하고 파싱하여 필요한 정보를 추출하며, 대화 기록을 세션 상태에 저장하여 지속적인 대화 흐름을 유지하고 필요 시 초기화할 수 있습니다.
- **모델 관리**: Hugging Face Hub를 통해 모델과 토크나이저를 관리하고 공유하여 지속적인 업데이트와 협업이 가능합니다.

## 데모 설명 및 구현 방법
- Colab 환경을 활용해 구현했습니다. Colab에 업로드 후, 위에서부터 하나씩 실행한 후 streamlit 데모를 구현할 수 있습니다. 이때,
```bash
import urllib
print("Password/Endpoint IP:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip("\n"))
```

을 통해 받은 IP 주소를 복사한 후,

```bash
!npx localtunnel --port
```
를 통해 연결된 링크에서 IP 주소를 복사하면 데모를 활용할 수 있습니다.
  
- Solar API를 활용해 1) Finetuning 및 2) RAG(Solar Embeddings) + Solar 성능을 비교하였고, 최종적으로
  2) RAG + Solar 챗봇을 선택하였습니다.
- 현재 제공된 py 파일에 api_key를 입력한 후, 순서에 따라 실행하면 데모를 확인할 수 있습니다.


구체적인 내용은 다음과 같습니다.
##
## RAG 관련
### 설치 방법

RAG 모델을 설치하고 실행하기 위해 아래의 단계를 따르세요:


1. **필요한 라이브러리 설치**:
   - 필요한 Python 라이브러리를 설치합니다. 터미널이나 명령 프롬프트에서 다음 명령어를 실행하세요:

     ```bash
     !pip install langchain-pinecone langchain openai pinecone-client streamlit
     ```

   - 이 명령어는 `langchain`, `pinecone-client`, `openai`, `streamlit` 등 RAG 모델 실행과 웹 인터페이스 구성을 위한 필수 라이브러리를 설치합니다.

2. **API 키 설정**:
   - Upstage와 Pinecone API 키를 설정합니다. 이 키들은 외부 서비스와의 통신을 위해 필요합니다.

     ```python
     UPSTAGE_API_KEY = "your-upstage-api-key"
     PINECONE_API_KEY = "your-pinecone-api-key"
     ```

3. **임베딩 및 벡터 저장소 설정**:
   - `UpstageEmbeddings`를 사용하여 문서와 질문을 벡터로 변환하고, `PineconeVectorStore`에 저장합니다.

     ```python
     from langchain_pinecone import PineconeVectorStore
     from langchain_upstage import UpstageEmbeddings

     embeddings = UpstageEmbeddings(model="solar-embedding-1-large-query", upstage_api_key=UPSTAGE_API_KEY)
     db = PineconeVectorStore(pinecone_api_key=PINECONE_API_KEY, index_name="upstage", embedding=embeddings, text_key='chunk')
     ```

4. **RAG 체인 초기화**:
   - `RetrievalQA` 체인을 구성하여 문서에서 관련 정보를 검색하고 답변을 생성하도록 설정합니다.

     ```python
     from langchain.chains import RetrievalQA
     from langchain_upstage import ChatUpstage

     qa_chain = RetrievalQA.from_chain_type(
         llm=ChatUpstage(api_key=UPSTAGE_API_KEY, model="solar-pro", max_tokens=2000),
         chain_type="stuff",
         retriever=db.as_retriever(search_kwargs={"k": 3}),
         return_source_documents=True
     )
     ```

5. **Streamlit 웹 인터페이스 구현**:
   - Streamlit을 사용하여 사용자 친화적인 웹 인터페이스를 구축하고, 문서를 업로드하고 질문을 입력할 수 있도록 합니다.

     ```python
     import streamlit as st

     st.title("ESG Converting Chatbot")
     uploaded_file = st.file_uploader("Upload a document (.txt, .md, .pdf)", type=("txt", "md", "pdf"))

     if uploaded_file:
         # 문서 파싱 및 질문 응답 처리
         # ...
         question = st.text_area("질문을 입력하세요:", placeholder="궁금하신 ESRS나 GRI 지표에 대해 질문하세요!")
         if question:
             response = ask_question(qa_chain, question)
             st.markdown("### Answer:")
             st.markdown(response)
     ```

### 사용 방법
1. **모델 및 토크나이저 로드**:
   - 사전 학습된 모델과 토크나이저를 로드합니다.

     ```python
     from transformers import AutoModelForQuestionAnswering, AutoTokenizer

     model = AutoModelForQuestionAnswering.from_pretrained("easyoon/finetuned_model")
     tokenizer = AutoTokenizer.from_pretrained("easyoon/llminno_tokenizer")
     ```

2. **데이터 전처리 및 토큰화**:
   - 입력 데이터를 전처리하고 토큰화합니다.

     ```python
     input_text = "ESRS에 대해 알려줘"
     context = "여기에 ESRS와 관련된 문서나 텍스트를 입력하세요."

     inputs = tokenizer(input_text, context, return_tensors="pt")
     ```

3. **모델 예측 수행**:
   - 준비된 데이터를 모델에 입력하여 예측을 수행합니다.

     ```python
     outputs = model(**inputs)

     start_logits = outputs.start_logits
     end_logits = outputs.end_logits

     # 결과 해석
     answer_start = torch.argmax(start_logits)
     answer_end = torch.argmax(end_logits) + 1
     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][answer_start:answer_end]))

     print("Answer:", answer)
     ```

4. **결과 출력**:
   - 모델의 예측 결과를 해석하여 사용자에게 답변을 제공합니다.

### 사용 방법

1. **문서 업로드 및 파싱**:
   - Streamlit 웹 인터페이스를 통해 문서를 업로드합니다. 지원되는 형식은 `.txt`, `.md`, `.pdf`입니다.

     ```python
     import streamlit as st

     uploaded_file = st.file_uploader("Upload a document (.txt, .md, .pdf)", type=("txt", "md", "pdf"))
     ```

   - 업로드된 문서는 파싱되어 필요한 정보가 추출됩니다.

2. **RAG 체인 초기화**:
   - 파싱된 문서를 기반으로 RAG 체인을 초기화합니다.

     ```python
     from ragmodel import initialize_rag

     if uploaded_file:
         parsed_text = parse_document(uploaded_file)
         qa_chain = initialize_rag(parsed_text)
     ```

3. **질문 입력 및 답변 생성**:
   - 사용자로부터 질문을 입력받고, RAG 모델을 통해 답변을 생성합니다.

     ```python
     question = st.text_area("질문을 입력하세요:", placeholder="궁금하신 ESRS나 GRI 지표에 대해 질문하세요!")

     if question:
         response = ask_question(qa_chain, question)
         st.markdown("### Answer:")
         st.markdown(response)
     ```

4. **결과 출력**:
   - 모델이 생성한 답변을 사용자에게 제공합니다.

## Finetuning 관련 설치

### 설치 방법

Fine-Tuning 모델을 설치하고 실행하기 위해 아래의 단계를 따르세요:

1. **필요한 라이브러리 설치**:
   - 먼저, 필요한 Python 라이브러리를 설치합니다. 터미널이나 명령 프롬프트에서 다음 명령어를 실행하세요:

     ```bash
     !pip install transformers datasets torch peft
     ```

   - 이 명령어는 `transformers`, `datasets`, `torch`, `peft` 등 모델 훈련과 데이터 처리를 위한 필수 라이브러리를 설치합니다.

2. **사전 학습된 모델 및 데이터셋 로드**:
   - Hugging Face의 `transformers` 라이브러리를 사용하여 사전 학습된 모델을 로드합니다. BERT 모델을 사용하려면 다음과 같이 코드를 작성하세요:

     ```python
     from transformers import AutoModelForQuestionAnswering, AutoTokenizer

     model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
     model = AutoModelForQuestionAnswering.from_pretrained(model_name)
     tokenizer = AutoTokenizer.from_pretrained(model_name)
     ```

   - `datasets` 라이브러리를 사용하여 도메인에 맞는 데이터셋을 로드합니다.

3. **데이터 전처리 및 토크나이저 설정**:
   - 데이터를 모델에 맞게 전처리합니다.

     ```python
     def clean_text_columns(df, columns):
         for col in columns:
             df[col + '_cleaned'] = (
                 df[col]
                 .astype(str)
                 .str.replace("\t|\n", " ")
                 .str.replace(r" {2,}", " ", regex=True)
                 .str.replace(r"[\*\-,|]", "", regex=True)
                 .str.strip()
             )
         return df

     # 데이터셋 전처리
     df = clean_text_columns(df, ['example'])
     ```

   - 토크나이저를 사용하여 데이터를 토큰화합니다.

4. **훈련 설정 및 실행**:
   - `Trainer` 클래스를 사용하여 훈련 인자를 설정하고, 모델을 미세 조정합니다.

     ```python
     from transformers import Trainer, TrainingArguments

     training_args = TrainingArguments(
         output_dir="./results",
         evaluation_strategy="epoch",
         learning_rate=2e-5,
         per_device_train_batch_size=8,
         per_device_eval_batch_size=8,
         num_train_epochs=3,
         weight_decay=0.01,
     )

     trainer = Trainer(
         model=model,
         args=training_args,
         train_dataset=tokenized_train_dataset,
         eval_dataset=tokenized_eval_dataset,
     )
     
     trainer.train()
     ```

5. **모델 평가 및 저장**:
   - 훈련된 모델을 평가하고, 필요에 따라 저장하여 나중에 사용할 수 있도록 합니다.

     ```python
     trainer.save_model('./finetuned_model')
     ```

### 설치 방법

1. **필요한 라이브러리 설치**:
   - 먼저, 필요한 Python 라이브러리를 설치합니다. 터미널이나 명령 프롬프트에서 다음 명령어를 실행하세요:

     ```bash
     !pip install transformers datasets torch
     ```

   - 이 명령어는 `transformers`, `datasets`, `torch` 등 모델 실행과 데이터 처리를 위한 필수 라이브러리를 설치합니다.

2. **사전 학습된 모델 및 토크나이저 로드**:
   - Hugging Face의 `transformers` 라이브러리를 사용하여 사전 학습된 모델을 로드합니다.
     ```python
     from transformers import AutoModelForQuestionAnswering, AutoTokenizer

     model = AutoModelForQuestionAnswering.from_pretrained("easyoon/finetuned_model")
     tokenizer = AutoTokenizer.from_pretrained("easyoon/llminno_tokenizer")
     ```

3. **데이터 전처리 및 토큰화**:
   - 데이터를 모델에 맞게 전처리하고 토큰화합니다.

     ```python
     def clean_text_columns(df, columns):
         for col in columns:
             df[col + '_cleaned'] = (
                 df[col]
                 .astype(str)
                 .str.replace("\t|\n", " ")
                 .str.replace(r" {2,}", " ", regex=True)
                 .str.replace(r"[\*\-,|]", "", regex=True)
                 .str.strip()
             )
         return df

     # 데이터셋 전처리
     df = clean_text_columns(df, ['example'])
     ```

   - 토크나이저를 사용하여 데이터를 토큰화합니다.

4. **모델 예측 수행**:
   - 준비된 데이터를 사용하여 모델 예측을 수행합니다.

     ```python
     inputs = tokenizer("질문 텍스트", "문맥 텍스트", return_tensors="pt")
     outputs = model(**inputs)

     start_logits = outputs.start_logits
     end_logits = outputs.end_logits

     # 결과 출력
     print("Start logits:", start_logits)
     print("End logits:", end_logits)
     ```

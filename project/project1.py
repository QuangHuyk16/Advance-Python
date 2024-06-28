# Chatbot là một chương trình máy tính hiểu mục đích truy vấn của bạn để trả lời bằng giải pháp.
# Chatbots là ứng dụng phổ biến nhất của Xử lý ngôn ngữ tự nhiên trong ngành
# CHAT BOT ĐẦU CUỐI LÀ GÌ?
# Chatbot đầu cuối đề cập đến một chatbot có thể xử lý một cuộc trò chuyện hoàn chỉnh từ đầu đến cuối mà không cần sự trợ giúp của con người.
# Để tạo một chatbot end-to-end, bạn cần viết một chương trình máy tính có thể hiểu yêu cầu của người dùng, tạo ra phản hồi phù hợp và thực hiện hành động khi cần thiết. 
# Điều này bao gồm việc thu thập dữ liệu, chọn ngôn ngữ lập trình và các công cụ NLP, đào tạo chatbot cũng như kiểm tra và tinh chỉnh nó trước khi cung cấp cho người dùng. 
# Sau khi triển khai, người dùng có thể tương tác với chatbot bằng cách gửi cho nó nhiều yêu cầu và chatbot có thể tự xử lý toàn bộ cuộc trò chuyện.
# Để tạo một chatbot end-to-end bằng Python, chúng ta có thể làm theo các bước được đề cập bên dưới:
# 1.Xác định ý định
# 2.Tạo dữ liệu đào tạo
# 3.Đào tạo chatbot
# 4.Xây dựng chatbot
# 5.Kiểm tra chatbot
# 6.Triển khai chatbot




# Chatbot kết thúc bằng cách sử dụng Python
# os: Thư viện cung cấp các hàm tương tác với hệ điều hành.
# nltk: Thư viện xử lý ngôn ngữ tự nhiên.
# ssl: Được sử dụng để tạo SSL context để tải dữ liệu từ một nguồn không đáng tin cậy.
# streamlit: Thư viện dùng để xây dựng ứng dụng web nhanh chóng với Python.
# random: Thư viện cung cấp các hàm liên quan đến số ngẫu nhiên.
# TfidfVectorizer: Được sử dụng để chuyển đổi văn bản thành ma trận TF-IDF.
# LogisticRegression: Mô hình hồi quy logistic.
import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression



# Dòng đầu tiên thiết lập một SSL context không xác minh để tải dữ liệu từ nguồn không đáng tin cậy.
# Dòng thứ hai chỉ định thư mục để tải dữ liệu từ NLTK.
# Dòng thứ ba tải dữ liệu 'punkt' từ NLTK, được sử dụng cho phân tích cú pháp.
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')



# Mỗi phần tử trong danh sách intents đại diện cho một loại ý định (intention) mà người dùng có 
# thể có khi nói chuyện với chatbot. Mỗi ý định có các mẫu dữ liệu (patterns) mà người dùng có 
# thể nói và các phản hồi (responses) tương ứng từ chatbot.
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine, thank you", "Nothing much"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.", "Age is just a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.", "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    }
]




# Tạo vectorizer và phân loại
# TfidfVectorizer được sử dụng để chuyển đổi các mẫu dữ liệu thành ma trận TF-IDF.
# LogisticRegression là mô hình phân loại logistic được sử dụng để phân loại các mẫu dữ liệu.
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)



# Xử lý trước dữ liệu
# Duyệt qua các ý định và mẫu dữ liệu trong intents và tạo hai danh sách tags và patterns, 
# chứa nhãn của mỗi mẫu dữ liệu và các mẫu dữ liệu tương ứng.
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)



# training the model
# Sử dụng TfidfVectorizer để chuyển đổi patterns thành ma trận TF-IDF.
# Huấn luyện mô hình LogisticRegression để phân loại các mẫu dữ liệu.
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)



# Nhận một đoạn văn bản đầu vào từ người dùng và dự đoán nhãn của nó sử dụng mô hình đã huấn luyện.
# Chọn một phản hồi ngẫu nhiên từ danh sách các phản hồi tương ứng với nhãn dự đoán.
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response



# Tạo giao diện người dùng sử dụng Streamlit.
# Lặp lại việc lấy đầu vào từ người dùng và hiển thị phản hồi từ chatbot cho đến khi người dùng nhập "goodbye" hoặc "bye".
counter = 0
def main():
    global counter
    st.title("Chatbot is me")
    st.write("Chào mừng bạn đến với chatbot. Vui lòng nhập tin nhắn và nhấn Enter để bắt đầu cuộc trò chuyện với tôi.")

    counter += 1
    user_input = st.text_input("Bạn:", key=f"user_input_{counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ['goodbye', 'bye']:
            st.write("Cảm ơn bạn đã trò chuyện với tôi. Chúc bạn có một buổi học tuyệt tuyệt vời!")
            st.stop()

if __name__ == '__main__':
    main()






# Để chạy chatbot này, hãy sử dụng lệnh được đề cập bên dưới trong thiết bị đầu cuối của bạn:
# Streamlit run filename.py(copy path file) or python -m streamlit run filename.py(copy path file)
    
# Khi chạy ứng dụng Streamlit, đặc biệt là khi sử dụng lệnh streamlit run <filename.py> để chạy một tệp Python cụ thể, bạn cần đặt tệp Python đó trong một thư mục cụ thể. 
# Nguyên nhân chính là do Streamlit cần truy cập vào tệp Python và các tệp tài nguyên (như hình ảnh, CSS, và các tệp dữ liệu khác) mà ứng dụng của bạn có thể sử dụng.

# Khi bạn chạy lệnh streamlit run <filename.py>, Streamlit sẽ tự động phát hiện và tải các tệp tài nguyên được sử dụng bởi ứng dụng từ cùng thư mục hoặc các thư mục con của thư mục chứa tệp Python đó. 
# Do đó, để đảm bảo rằng Streamlit có thể tìm thấy tất cả các tệp cần thiết, đặt tệp Python và tất cả các tệp tài nguyên liên quan trong cùng một thư mục hoặc trong các thư mục con của thư mục đó là một cách tiếp cận phổ biến.

# Nếu tệp Python và các tệp tài nguyên không được đặt trong cùng một thư mục hoặc trong các thư mục con của nó, bạn có thể cung cấp đường dẫn tương đối đến tệp Python khi chạy lệnh streamlit run, nhưng điều này có thể gây khó khăn trong quản lý và triển khai ứng dụng của bạn.
# Do đó, việc đặt tất cả các tệp cần thiết trong cùng một thư mục hoặc trong các thư mục con của nó là phương pháp đơn giản và tiện lợi nhất.

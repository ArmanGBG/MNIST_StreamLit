import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# مدل ساده‌ی MNIST
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28*28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)

# بارگذاری مدل ذخیره‌شده
model = SimpleNN()
model.load_state_dict(torch.load("mnist_model.pth", map_location=torch.device("cpu")))
model.eval()

# تعریف ترنسفورم
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# رابط کاربری Streamlit
st.title("🧠 تشخیص عدد دست‌نویس")

uploaded_file = st.file_uploader("یک تصویر عدد آپلود کن", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='تصویر ورودی', use_column_width=True)

    image = image.convert("L")  # تبدیل به خاکستری
    image = transforms.functional.invert(image)  # اینورت کردن
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(1).item()

    st.success(f"🔢 عدد پیش‌بینی شده: {prediction}")

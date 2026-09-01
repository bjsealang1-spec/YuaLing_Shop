import streamlit as st
import random
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------
# 0. 페이지 설정 및 라벤더 감성 디자인 (YuaLing 샵)
# ---------------------------------------------------------
st.set_page_config(
    page_title="YuaLing Fashion Curation Shop",
    page_icon="💜",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #F9F6FC;
    }
    .stButton>button {
        background-color: #9B7EDE;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        padding: 0.6rem 1.2rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #8661DE;
        color: white;
    }
    h1, h2, h3 {
        color: #5D437C;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 1. PyTorch CNN 모델 정의 (FashionCNN)
# ---------------------------------------------------------
class FashionCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class_names = ["원피스", "스커트", "팬츠", "블라우스", "비키니/바캉스"]

@st.cache_resource
def load_trained_model():
    model = FashionCNN(num_classes=len(class_names))
    model.eval()
    return model

cnn_model = load_trained_model()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# ---------------------------------------------------------
# 2. 세션 상태 초기화 (미래 이름 '유아' 적용 버전!)
# ---------------------------------------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = "유아"
if 'wishlist' not in st.session_state:
    st.session_state.wishlist = []
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'order_history' not in st.session_state:
    st.session_state.order_history = []
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "🏠 홈"

if 'products_db' not in st.session_state:
    st.session_state.products_db = {
        # 원피스 라인
        "라벤더 실크 원피스": {"price": 45000, "category": "원피스", "rating": 4.9, "desc": "하늘하늘한 실루엣과 고급스러운 라벤더 컬러가 매력적인 데이트룩 원피스", "reviews": [{"text": "핏이 정말 공주님 같아요! 💜", "star": 5}]},
        "로맨틱 프릴 캉캉 원피스": {"price": 49000, "category": "원피스", "rating": 5.0, "desc": "걸을 때마다 살랑거리는 프릴 디테일이 청순함을 극대화해주는 캉캉 원피스", "reviews": [{"text": "입고 여행 갔는데 인생샷 건졌어요 ✨", "star": 5}]},
        "플로럴 샤링 미니 원피스": {"price": 42000, "category": "원피스", "rating": 4.8, "desc": "바디라인을 예쁘게 잡아주는 샤링 디테일과 화사한 플로럴 패턴의 조합", "reviews": [{"text": "여리여리해 보이고 너무 예뻐요!", "star": 5}]},
        "진주 카라 스퀘어 원피스": {"price": 52000, "category": "원피스", "rating": 4.9, "desc": "단아하면서도 고급스러운 진주 버튼과 스퀘어 넥라인의 하객룩 원피스", "reviews": [{"text": "단정하고 고급스러워 보여요.", "star": 5}]},
        "체크 트위드 미니 원피스": {"price": 56000, "category": "원피스", "rating": 4.7, "desc": "클래식한 트위드 소재와 세련된 체크 패턴으로 페미닌한 무드 완성", "reviews": [{"text": "재질도 도톰하고 핏이 딱 잡혀요!", "star": 5}]},

        # 블라우스 라인
        "리본 퍼프 블라우스": {"price": 36000, "category": "블라우스", "rating": 4.7, "desc": "청순한 퍼프 소매와 리본 포인트가 사랑스러운 데일리 블라우스", "reviews": [{"text": "얼굴이 화사해 보여요 ✨", "star": 5}]},
        "시스루 타이 블라우스": {"price": 38000, "category": "블라우스", "rating": 4.9, "desc": "우아한 타이 리본 디테일과 은은한 시스루로 여리여리한 무드 연출", "reviews": [{"text": "슬랙스랑 입으니 진심 고급스러워요.", "star": 5}]},
        "새틴 브이넥 블라우스": {"price": 39000, "category": "블라우스", "rating": 4.8, "desc": "은은한 광택감이 도는 고급 새틴 소재의 세련된 오피스룩 블라우스", "reviews": [{"text": "촉감이 너무 부드럽고 예뻐요.", "star": 5}]},
        "레이스 넥 프릴 블라우스": {"price": 34000, "category": "블라우스", "rating": 4.6, "desc": "넥라인 레이스 프릴이 사랑스러운 빈티지 무드의 블라우스", "reviews": [{"text": "레이어드해서 입기 최고예요!", "star": 5}]},

        # 스커트 라인
        "하이웨이스트 플리츠 스커트": {"price": 32000, "category": "스커트", "rating": 4.8, "desc": "다리가 길어 보이는 하이웨이스트 라인의 트렌디한 플리츠 스커트", "reviews": [{"text": "허리 밴딩이 편해서 자주 입어요~", "star": 5}]},
        "머메이드 롱 미디 스커트": {"price": 38000, "category": "스커트", "rating": 4.9, "desc": "곡선미를 예쁘게 살려주는 우아한 실루엣의 머메이드 스커트", "reviews": [{"text": "몸매가 엄청 예뻐 보여요 대박!", "star": 5}]},
        "체크 버튼 H라인 스커트": {"price": 33000, "category": "스커트", "rating": 4.7, "desc": "캐주얼하면서도 깔끔하게 떨어지는 핏의 H라인 스커트", "reviews": [{"text": "기장도 딱 좋고 날씬해 보여요.", "star": 5}]},

        # 팬츠/아우터 라인
        "시크 오버핏 블레이저": {"price": 58000, "category": "팬츠/아우터", "rating": 4.9, "desc": "툭 걸쳐도 세련된 무드가 살아나는 오버핏 데일리 블레이저", "reviews": [{"text": "오피스룩으로 완벽합니다!", "star": 5}]},
        "와이드 루즈핏 코튼 팬츠": {"price": 36000, "category": "팬츠/아우터", "rating": 4.8, "desc": "하루 종일 편안하게 입기 좋은 내추럴 핏 와이드 팬츠", "reviews": [{"text": "진짜 편해서 매일 입고 있어요.", "star": 5}]},
        "크롭 윈드브레이커 자켓": {"price": 49000, "category": "팬츠/아우터", "rating": 4.7, "desc": "간절기 환절기에 툭 걸치기 좋은 트렌디한 크롭 아우터", "reviews": [{"text": "디자인도 귀엽고 가벼워요!", "star": 5}]},

        # 비키니/바캉스 라인
        "글램 홀터넥 비키니": {"price": 42000, "category": "비키니/바캉스", "rating": 4.9, "desc": "바디라인을 슬림하고 볼륨감 있게 잡아주는 세련된 홀터넥 비키니", "reviews": [{"text": "컬러감도 고급스럽고 핏 대박이에요 🌊", "star": 5}]},
        "레이스 프릴 비치 원피스": {"price": 46000, "category": "비키니/바캉스", "rating": 5.0, "desc": "휴양지에서 로맨틱한 분위기를 완성해 주는 시스루 비치 커버업", "reviews": [{"text": "바닷가에서 사진 정말 잘 나와요!", "star": 5}]}
    }

def generate_free_fashion_styling(predicted_category):
    return "이 아이템은 트렌디한 무드가 가득해서 너만의 개성을 살려 멋지게 소화할 수 있을 거야! ✨"

def predict_fashion_image(img):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = cnn_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_idx = predicted.item()
    predicted_category = class_names[pred_idx]
    comment = generate_free_fashion_styling(predicted_category)
    return predicted_category, comment

# ---------------------------------------------------------
# 3. 사이드바 (로그인 및 관리자 메뉴)
# ---------------------------------------------------------
st.sidebar.markdown("# 💜 YuaLing Shop")
if not st.session_state.logged_in:
    input_name = st.sidebar.text_input("닉네임을 입력하세요", value="유아", key="sidebar_input_name")
    if st.sidebar.button("로그인하기", key="sidebar_login_btn"):
        st.session_state.logged_in = True
        st.session_state.user_name = input_name
        st.rerun()
else:
    st.sidebar.success(f"대표 **{st.session_state.user_name}**님 환영해요! 🎉")
    if st.sidebar.button("로그아웃", key="sidebar_logout_btn"):
        st.session_state.logged_in = False
        st.rerun()
    
    st.sidebar.markdown("---")
    if st.sidebar.button("👑 사장님 상품 등록실 열기", key="sidebar_admin_btn"):
        st.session_state.current_tab = "👑 상품등록"
        st.rerun()

# ---------------------------------------------------------
# 4. 에이블리 스타일 상단 배너 및 탭바 UI
# ---------------------------------------------------------
if not st.session_state.logged_in:
    st.title("💜 YuaLing Shop에 오신 것을 환영합니다!")
    st.markdown("사이드바에서 **로그인**을 진행해 주세요! ✨")
else:
    # --- [에이블리 스타일 상단 카테고리 퀵버튼 레이아웃] ---
    st.markdown("### 🛍️ YuaLing Hot Menu")
    q_cols = st.columns(5)
    with q_cols[0]:
        if st.button("🔥 첫구매 30%", key="top_q_first"):
            st.toast("🎉 첫구매 10% 쿠폰이 장바구니에 적용됩니다!")
    with q_cols[1]:
        if st.button("🌸 원피스특가", key="top_q_dress"):
            st.session_state.current_tab = "🏠 홈"
            st.session_state.selected_product = None
            st.rerun()
    with q_cols[2]:
        if st.button("✨ AI 큐레이터", key="top_q_ai"):
            st.session_state.current_tab = "✨ AI 큐레이터"
            st.rerun()
    with q_cols[3]:
        if st.button("🛒 장바구니", key="top_q_cart"):
            st.session_state.current_tab = "🛒 장바구니"
            st.rerun()
    with q_cols[4]:
        if st.button("👤 마이페이지", key="top_q_my"):
            st.session_state.current_tab = "👤 마이페이지"
            st.rerun()

    st.markdown("---")

    # --- [메인 콘텐츠 분기] ---
    tab = st.session_state.current_tab

    if tab == "🏠 홈":
        st.title("💜 YuaLing Shop 메인")
        
        st.info("🎁 **[EVENT]** 지금 'YUALING2026' 쿠폰을 쓰면 전 상품 10% 할인!")

        if st.session_state.selected_product:
            p_name = st.session_state.selected_product
            p_info = st.session_state.products_db[p_name]

            if st.button("⬅️ 목록으로 돌아가기", key="home_back_btn"):
                st.session_state.selected_product = None
                st.rerun()

            st.subheader(f"🏷️ {p_name}")
            st.caption(f"카테고리: {p_info['category']} | 평점: ⭐️ {p_info['rating']}")
            st.markdown(f"### 가격: ₩{p_info['price']:,}")
            st.write(f"**📝 상품 설명:** {p_info['desc']}")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🛒 장바구니 담기", key="home_detail_cart"):
                    st.session_state.cart.append({"name": p_name, "price": p_info['price']})
                    st.toast(f"'{p_name}'이(가) 장바구니에 담겼어요! 🛒")
            with c2:
                if st.button("❤️ 찜하기", key="home_detail_wish"):
                    item_dict = {"category": p_info['category'], "comment": f"{p_name} - {p_info['desc']}"}
                    if item_dict not in st.session_state.wishlist:
                        st.session_state.wishlist.append(item_dict)
                        st.toast("찜목록에 추가되었습니다! ❤️")
                    else:
                        st.warning("이미 찜한 상품이에요!")
        else:
            categories = ["전체", "원피스", "블라우스", "스커트", "팬츠/아우터", "비키니/바캉스"]
            selected_cat = st.selectbox("원하는 카테고리를 선택해 보세요 ✨", categories, key="home_cat_select")

            if selected_cat == "전체":
                filtered_items = list(st.session_state.products_db.keys())
            else:
                filtered_items = [name for name, data in st.session_state.products_db.items() if selected_cat in data['category']]

            cols = st.columns(min(len(filtered_items), 3) if len(filtered_items) > 0 else 1)
            for idx, item_name in enumerate(filtered_items):
                info = st.session_state.products_db[item_name]
                col = cols[idx % len(cols)]
                with col:
                    st.markdown(f"**{item_name}**")
                    st.caption(f"⭐️ {info['rating']} | ₩{info['price']:,}")
                    if st.button(f"상세보기 🔍", key=f"filter_item_{idx}"):
                        st.session_state.selected_product = item_name
                        st.rerun()

    elif tab == "✨ AI 큐레이터":
        st.title("💜 YuaLing PyTorch CNN 스타일 큐레이터")
        uploaded_file = st.file_uploader("패션 이미지를 업로드해 주세요 (PNG, JPG)", type=["png", "jpg", "jpeg"], key="ai_file_uploader")
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="업로드한 아이템", width=300)
            if st.button("✨ CNN 스타일 분석 시작하기", key="ai_predict_btn"):
                with st.spinner("딥러닝 모델 추론 중... ⏳"):
                    category, comment = predict_fashion_image(image)
                st.success(f"**🏷️ 예측 카테고리:** {category}")
                st.info(f"**💬 추천 코멘트:** {comment}")

    elif tab == "🛒 장바구니":
        st.title(f"🛒 {st.session_state.user_name}님의 장바구니")
        if len(st.session_state.cart) == 0:
            st.info("장바구니가 비어있어요!")
        else:
            total_price = sum([item['price'] for item in st.session_state.cart])
            for cart_item in st.session_state.cart:
                st.markdown(f"- **{cart_item['name']}** : ₩{cart_item['price']:,}")
            
            coupon = st.text_input("쿠폰 코드 입력", value="YUALING2026", key="cart_coupon_input")
            final_price = int(total_price * 0.9) if coupon == "YUALING2026" else total_price
            
            st.markdown(f"### 💳 총 결제 금액: **₩{final_price:,}**")
            if st.button("🛍️ 주문하기", key="cart_order_btn"):
                st.balloons()
                st.success("주문이 완료되었습니다! 💜")
                st.session_state.order_history.append({"items": list(st.session_state.cart), "total": final_price})
                st.session_state.cart = []

    elif tab == "👤 마이페이지":
        st.title(f"👤 {st.session_state.user_name}님의 마이페이지")
        tab1, tab2 = st.tabs(["❤️ 찜목록", "📦 주문 내역"])
        with tab1:
            if not st.session_state.wishlist:
                st.info("찜한 상품이 없어요.")
            for saved in st.session_state.wishlist:
                st.write(saved['comment'])
        with tab2:
            if not st.session_state.order_history:
                st.info("주문 내역이 없어요.")
            for order in st.session_state.order_history:
                st.write(order)

    elif tab == "👑 상품등록":
        st.title("👑 YuaLing 대표님 상품 등록실")
        new_p_name = st.text_input("상품 이름", key="reg_name_input")
        new_p_price = st.number_input("가격", value=35000, key="reg_price_input")
        new_p_cat = st.selectbox("카테고리", ["원피스", "블라우스", "비키니/바캉스", "스커트", "팬츠/아우터"], key="reg_cat_select")
        new_p_desc = st.text_area("설명", key="reg_desc_input")
        if st.button("🚀 신상 등록하기", key="reg_submit_btn"):
            st.session_state.products_db[new_p_name] = {"price": new_p_price, "category": new_p_cat, "rating": 5.0, "desc": new_p_desc, "reviews": []}
            st.success("등록 완료! 🎉")

    # --- [하단 고정 모바일 링크 탭바] ---
    st.markdown("---")
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    with b_col1:
        if st.button("🏠 홈으로", key="bot_nav_home"):
            st.session_state.current_tab = "🏠 홈"
            st.session_state.selected_product = None
            st.rerun()
    with b_col2:
        if st.button("✨ AI큐레이터", key="bot_nav_ai"):
            st.session_state.current_tab = "✨ AI 큐레이터"
            st.rerun()
    with b_col3:
        if st.button("🛒 장바구니 보기", key="bot_nav_cart"):
            st.session_state.current_tab = "🛒 장바구니"
            st.rerun()
    with b_col4:
        if st.button("👤 마이페이지", key="bot_nav_my"):
            st.session_state.current_tab = "👤 마이페이지"
            st.rerun()
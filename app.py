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
        # 이미지 크기 64x64 기준 전결합층 (FC)
        self.fc = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

# 모델 및 클래스 라벨 설정
class_names = ["원피스", "스커트", "팬츠", "블라우스", "비키니/바캉스"]

@st.cache_resource
def load_trained_model():
    # 모델 인스턴스 생성 (실제 학습된 가중치가 있다면 .load_state_dict()로 로드 가능)
    model = FashionCNN(num_classes=len(class_names))
    model.eval()
    return model

cnn_model = load_trained_model()

# 이미지 전처리 파이프라인
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# ---------------------------------------------------------
# 2. 세션 상태 초기화 (유아 대표님 로그인 & 장바구니/찜목록/주문내역)
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

# 상품 마스터 데이터베이스
if 'products_db' not in st.session_state:
    st.session_state.products_db = {
        # --- 원피스 6종 ---
        "라벤더 실크 원피스": {
            "price": 45000, "category": "원피스", "rating": 4.9, 
            "desc": "하늘하늘한 실루엣과 고급스러운 라벤더 컬러가 매력적인 데이트룩 원피스", 
            "reviews": [{"text": "핏이 정말 공주님 같아요! 💜", "star": 5}, {"text": "색감이 미쳤습니다 강력 추천!", "star": 5}]
        },
        "로맨틱 프릴 캉캉 원피스": {
            "price": 49000, "category": "원피스", "rating": 5.0, 
            "desc": "걸을 때마다 살랑거리는 프릴 디테일이 청순함을 극대화해주는 캉캉 원피스", 
            "reviews": [{"text": "입고 여행 갔는데 인생샷 건졌어요 ✨", "star": 5}]
        },
        "플로럴 샤링 미니 원피스": {
            "price": 42000, "category": "원피스", "rating": 4.8, 
            "desc": "바디라인을 예쁘게 잡아주는 샤링 디테일과 화사한 플로럴 패턴의 조합", 
            "reviews": [{"text": "여리여리해 보이고 너무 예뻐요!", "star": 5}]
        },
        "파스텔 민트 니트 원피스": {
            "price": 46000, "category": "원피스", "rating": 4.7, 
            "desc": "부드러운 촉감과 은은한 파스텔 민트 컬러로 단아한 매력을 주는 니트 원피스", 
            "reviews": [{"text": "까슬거리지 않고 편하게 입기 좋아요.", "star": 5}]
        },
        "시크 블랙 미니 원피스": {
            "price": 52000, "category": "원피스", "rating": 4.9, 
            "desc": "군더더기 없이 깔끔하게 떨어지는 핏으로 세련된 무드를 주는 블랙 원피스", 
            "reviews": [{"text": "하객룩으로 입었는데 다들 어디서 샀냐고 물어봐요!", "star": 5}]
        },
        "진주 버튼 트위드 원피스": {
            "price": 62000, "category": "원피스", "rating": 5.0, 
            "desc": "고급스러운 트위드 원단에 반짝이는 진주 버튼으로 우아함을 더한 프리미엄 원피스", 
            "reviews": [{"text": "실물로 보면 진짜 백화점 퀄리티예요 🤍", "star": 5}]
        },
        # --- 블라우스 6종 ---
        "리본 퍼프 블라우스": {
            "price": 36000, "category": "블라우스", "rating": 4.7, 
            "desc": "청순한 퍼프 소매와 리본 포인트가 사랑스러운 데일리 블라우스", 
            "reviews": [{"text": "얼굴이 화사해 보여요 ✨", "star": 5}]
        },
        "시스루 타이 블라우스": {
            "price": 38000, "category": "블라우스", "rating": 4.9, 
            "desc": "우아한 타이 리본 디테일과 은은한 시스루로 여리여리한 무드 연출", 
            "reviews": [{"text": "슬랙스랑 입으니 진심 고급스러워요.", "star": 5}]
        },
        "레이스 스퀘어넥 블라우스": {
            "price": 35000, "category": "블라우스", "rating": 4.8, 
            "desc": "목선과 쇄골라인을 예쁘게 드러내 주는 페미닌한 레이스 스퀘어넥 블라우스", 
            "reviews": [{"text": "핏이 단정하면서도 여성스러워요!", "star": 5}]
        },
        "새틴 실크 타이즈 블라우스": {
            "price": 41000, "category": "블라우스", "rating": 4.9, 
            "desc": "은은한 광택감이 감도는 부드러운 새틴 소재의 고급스러운 블라우스", 
            "reviews": [{"text": "만지는 촉감도 좋고 입었을 때 확 기품 있어 보여요.", "star": 5}]
        },
        "시폰 프릴 카라 블라우스": {
            "price": 37000, "category": "블라우스", "rating": 4.7, 
            "desc": "하늘하늘한 시폰 소재와 러블리한 프릴 카라가 매력적인 페미닌 블라우스", 
            "reviews": [{"text": "스커트에 쏙 넣어 입으면 진짜 여신 룩 완성!", "star": 5}]
        },
        "모던 브이넥 실키 블라우스": {
            "price": 39000, "category": "블라우스", "rating": 4.8, 
            "desc": "깔끔한 브이넥 라인으로 시크하면서도 오피스룩으로 제격인 베이직 블라우스", 
            "reviews": [{"text": "데일리 출근룩으로 매일 입고 있어요 최고!", "star": 5}]
        },
        # --- 비키니/바캉스 6종 ---
        "글램 홀터넥 비키니": {
            "price": 42000, "category": "비키니/바캉스", "rating": 4.9, 
            "desc": "바디라인을 슬림하고 볼륨감 있게 잡아주는 세련된 홀터넥 비키니", 
            "reviews": [{"text": "컬러감도 고급스럽고 핏 대박이에요 🌊", "star": 5}]
        },
        "로맨틱 플로럴 비키니": {
            "price": 39000, "category": "비키니/바캉스", "rating": 4.8, 
            "desc": "여리여리한 플로럴 패턴으로 바캉스 분위기를 물씬 풍기는 스위트 비키니", 
            "reviews": [{"text": "하와이 여행 가서 너무 잘 입었어요!", "star": 5}]
        },
        "트로피컬 하이웨이스트 비키니": {
            "price": 44000, "category": "비키니/바캉스", "rating": 4.9, 
            "desc": "다리가 길어 보이고 체형 커버까지 완벽하게 도와주는 트로피컬 비키니", 
            "reviews": [{"text": "하이웨이스트라 노출 부담도 없고 군살 다 가려줘요!", "star": 5}]
        },
        "심플 스트랩 튜브탑 비키니": {
            "price": 38000, "category": "비키니/바캉스", "rating": 4.7, 
            "desc": "깔끔하고 모던한 무드로 시선을 사로잡는 오프숄더 스타일 튜브탑 비키니", 
            "reviews": [{"text": "어깨 라인이 엄청 여리여리해 보여요 대만족!", "star": 5}]
        },
        "볼륨 프릴 오프숄더 비키니": {
            "price": 45000, "category": "비키니/바캉스", "rating": 5.0, 
            "desc": "풍성한 프릴 디테일로 귀여우면서도 볼륨감을 살려주는 오프숄더 비키니", 
            "reviews": [{"text": "실물 색감이 휴양지랑 완전 찰떡이에요 💛", "star": 5}]
        },
        "시스루 로브 3P 비키니 세트": {
            "price": 54000, "category": "비키니/바캉스", "rating": 4.9, 
            "desc": "비키니와 함께 우아하게 걸치기 좋은 시스루 로브가 포함된 세트 상품", 
            "reviews": [{"text": "로브까지 세트인데 이 가격 대박 실화인가요?", "star": 5}]
        },
        # --- 기타 인기 상품들 ---
        "하이웨이스트 플리츠 스커트": {
            "price": 32000, "category": "스커트", "rating": 4.8, 
            "desc": "다리가 길어 보이는 하이웨이스트 라인의 트렌디한 플리츠 스커트", 
            "reviews": [{"text": "허리 밴딩이 편해서 자주 입어요~", "star": 5}, {"text": "코디하기 너무 쉬워요!", "star": 4}]
        },
        "시크 오버핏 블레이저": {
            "price": 58000, "category": "팬츠/아우터", "rating": 4.9, 
            "desc": "툭 걸쳐도 세련된 무드가 살아나는 오버핏 데일리 블레이저", 
            "reviews": [{"text": "어깨 패드가 과하지 않고 예뻐요.", "star": 5}, {"text": "오피스룩으로 완벽합니다!", "star": 5}]
        },
        "와이드 빈티지 데님 팬츠": {
            "price": 39000, "category": "팬츠/아우터", "rating": 4.8, 
            "desc": "내추럴한 워싱과 완벽한 핏을 자랑하는 힙한 와이드 데님", 
            "reviews": [{"text": "스트릿 감성 대박 편해요!", "star": 5}]
        }
    }

# ---------------------------------------------------------
# 3. 스타일리스트 코멘트 생성 및 CNN 모델 예측 함수
# ---------------------------------------------------------
def generate_free_fashion_styling(predicted_category):
    styling_tips = {
        "원피스": [
            "하늘하늘한 실루엣이 정말 매력적인 원피스야! 🌸 가벼운 가디건이나 심플한 토트백을 매치하면 데이트룩으로 완벽할 거야.",
            "페미닌한 무드를 극대화해주는 원피스네✨ 클래식한 메리제인 슈즈나 숏자켓을 걸쳐주면 스타일리시함이 두 배가 될 거야!"
        ],
        "스커트": [
            "트렌디한 핏이 돋보이는 스커트야! 👗 상의는 심플한 무지 티셔츠나 셔츠를 깔끔하게 넣어 입으면 포인트를 살릴 수 있어.",
            "발랄하면서도 세련된 느낌을 주는 스커트네 💫 캐주얼한 스니커즈나 롱삭스로 마무리하면 스트릿 룩 완성!"
        ],
        "팬츠": [
            "활동성과 스타일을 모두 잡은 멋스러운 팬츠야! 👖 루즈핏 셔츠나 크롭탑과 함께 입으면 요즘 감성에 딱 맞지.",
            "군더더기 없이 떨어지는 핏이 아주 멋진 팬츠네 😎 깔끔한 블레이저를 툭 걸쳐주면 시크한 오피스룩으로도 활용 만점이야!"
        ],
        "블라우스": [
            "우아한 디테일이 살아있는 블라우스야! 👚 슬랙스나 데님 팬츠 어디에나 찰떡같이 어울려서 활용도가 정말 높아.",
            "화사한 얼굴빛을 만들어주는 블라우스네 ✨ 진주 목걸이나 미니멀한 악세서리를 더해주면 훨씬 고급스러워 보여!"
        ],
        "비키니/바캉스": [
            "시선 집중! 🌊 휴양지나 해변에서 주인공으로 만들어줄 매력 만점 바캉스룩이야. 로브나 린넨셔츠를 걸쳐주면 스타일링 완성!",
            "청량하면서도 트렌디한 무드가 물씬 풍기는 비키니네 ☀️ 선글라스랑 라피아 햇을 더해주면 완벽한 휴가 패션이 될 거야!"
        ]
    }
    tips = styling_tips.get(predicted_category, ["정말 유니크하고 예쁜 패션 아이템이야! ✨ 너만의 개성을 살려서 멋지게 소화해 봐."])
    return random.choice(tips)

def predict_fashion_image(img):
    # PyTorch CNN 모델 추론 과정
    img_tensor = transform(img).unsqueeze(0) # 배치 차원 추가 [1, 3, 64, 64]
    
    with torch.no_grad():
        outputs = cnn_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        pred_idx = predicted.item()
    
    predicted_category = class_names[pred_idx]
    comment = generate_free_fashion_styling(predicted_category)
    return predicted_category, comment

# ---------------------------------------------------------
# 4. 사이드바 (YuaLing 샵 네비게이션)
# ---------------------------------------------------------
st.sidebar.markdown("# 💜 YuaLing Shop")

if not st.session_state.logged_in:
    st.sidebar.markdown("### 🔐 로그인 필요")
    input_name = st.sidebar.text_input("닉네임을 입력하세요", value="유아")
    if st.sidebar.button("로그인하기"):
        st.session_state.logged_in = True
        st.session_state.user_name = input_name
        st.rerun()
    st.sidebar.info("로그인하고 유아링 샵의 다양한 서비스를 즐겨보세요! ✨")
else:
    st.sidebar.success(f"대표 **{st.session_state.user_name}**님 환영해요! 🎉")
    if st.sidebar.button("로그아웃"):
        st.session_state.logged_in = False
        st.session_state.wishlist = []
        st.session_state.cart = []
        st.session_state.selected_product = None
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("🛍️ 쇼핑몰 메뉴")

    menu = st.sidebar.radio(
        "이동할 메뉴:", 
        ["🏠 홈 (메인 화면)", "🔍 상품 검색하기", "🛒 장바구니", "✨ AI 스타일 큐레이터", "📖 시즌 스타일 가이드", "👑 사장님 상품 관리", "👤 마이 페이지"]
    )

# ---------------------------------------------------------
# 5. 메인 화면 및 각 페이지별 콘텐츠
# ---------------------------------------------------------
if not st.session_state.logged_in:
    st.title("💜 YuaLing Shop에 오신 것을 환영합니다!")
    st.markdown("사이드바에서 **로그인**을 진행해 주세요! ✨")
else:
    if menu == "🏠 홈 (메인 화면)":
        st.title("💜 YuaLing Shop Curation Mall")
        st.markdown("### 🔥 2026 S/S 유아링 단독 기획전 & 트렌드 메인")
        st.info(f"✨ **[WELCOME]** {st.session_state.user_name} 대표님의 센스가 빛나는 트렌디한 공간입니다 🛍️")

        st.markdown("---")

        if st.session_state.selected_product:
            p_name = st.session_state.selected_product
            p_info = st.session_state.products_db[p_name]

            if st.button("⬅️ 홈으로 돌아가기"):
                st.session_state.selected_product = None
                st.rerun()

            st.subheader(f"🏷️ {p_name}")
            st.caption(f"카테고리: {p_info['category']} | 평점: ⭐️ {p_info['rating']}")
            st.markdown(f"### 가격: ₩{p_info['price']:,}")
            st.write(f"**📝 상품 설명:** {p_info['desc']}")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🛒 장바구니 담기"):
                    st.session_state.cart.append({"name": p_name, "price": p_info['price']})
                    st.toast(f"'{p_name}'이(가) 장바구니에 담겼어요! 🛒")
            with c2:
                if st.button("❤️ 찜하기"):
                    item_dict = {"category": p_info['category'], "comment": f"{p_name} - {p_info['desc']}"}
                    if item_dict not in st.session_state.wishlist:
                        st.session_state.wishlist.append(item_dict)
                        st.toast("찜목록에 추가되었습니다! ❤️")
                    else:
                        st.warning("이미 찜한 상품이에요!")

            st.markdown("---")
            st.subheader("💬 구매자 생생 리뷰")
            for rev in p_info['reviews']:
                stars = "⭐️" * rev['star']
                st.info(f"{stars}  \"{rev['text']}\"")

            st.markdown("#### ✨ 나도 한 줄 리뷰 남기기")
            new_review = st.text_input("리뷰 내용을 입력해주세요", key="rev_input")
            new_star = st.selectbox("별점 선택", [5, 4, 3, 2, 1], format_func=lambda x: "⭐️" * x, key="star_input")

            if st.button("리뷰 등록하기"):
                if new_review:
                    p_info['reviews'].append({"text": new_review, "star": new_star})
                    st.success("소중한 리뷰가 등록되었습니다! 🎉")
                    st.rerun()
                else:
                    st.warning("리뷰 내용을 입력해주세요.")
        else:
            st.subheader("🛍️ 카테고리별 쇼핑")
            categories = ["전체", "원피스", "블라우스", "비키니/바캉스", "스커트", "팬츠/아우터"]
            selected_cat = st.selectbox("보고 싶은 카테고리를 선택해 보세요 ✨", categories)

            st.markdown("---")

            if selected_cat == "전체":
                filtered_items = list(st.session_state.products_db.keys())
            else:
                filtered_items = [name for name, data in st.session_state.products_db.items() if selected_cat in data['category']]

            if len(filtered_items) == 0:
                st.info("해당 카테고리에 등록된 상품이 없어요!")
            else:
                cols = st.columns(min(len(filtered_items), 3))
                for idx, item_name in enumerate(filtered_items):
                    info = st.session_state.products_db[item_name]
                    col = cols[idx % len(cols)]
                    with col:
                        st.markdown(f"**{item_name}**")
                        st.caption(f"⭐️ {info['rating']} | {info['category']}")
                        st.markdown(f"**₩{info['price']:,}**")
                        if st.button(f"상세보기 🔍", key=f"filter_{idx}"):
                            st.session_state.selected_product = item_name
                            st.rerun()

    elif menu == "🔍 상품 검색하기":
        st.title("🔍 YuaLing 통합 상품 검색")
        st.markdown("원하는 키워드를 검색해 보세요!")
        search_query = st.text_input("검색어를 입력하세요", placeholder="원피스, 블라우스, 비키니 등...")
        if search_query:
            st.subheader(f"🔎 '{search_query}' 검색 결과")
            matched = [name for name, data in st.session_state.products_db.items() if search_query in name or search_query in data['category'] or search_query in data['desc']]
            if len(matched) == 0:
                st.warning("검색 결과가 없어요 다른 키워드로 검색해 보세요!")
            else:
                for match_name in matched:
                    m_info = st.session_state.products_db[match_name]
                    with st.expander(f"✨ {match_name} (₩{m_info['price']:,})"):
                        st.write(m_info['desc'])
                        if st.button("장바구니 담기", key=f"search_cart_{match_name}"):
                            st.session_state.cart.append({"name": match_name, "price": m_info['price']})
                            st.toast("장바구니에 담겼습니다! 🛒")

    elif menu == "🛒 장바구니":
        st.title(f"🛒 {st.session_state.user_name}님의 장바구니")
        st.markdown("담아둔 상품들을 확인하고 결제할 수 있는 공간이에요 ✨")
        if len(st.session_state.cart) == 0:
            st.info("장바구니가 비어있어요! 홈 화면에서 마음에 드는 상품을 담아보세요.")
        else:
            total_price = 0
            for i, cart_item in enumerate(st.session_state.cart):
                st.markdown(f"- **{cart_item['name']}** : ₩{cart_item['price']:,}")
                total_price += cart_item['price']

            st.markdown("---")
            coupon = st.text_input("할인 쿠폰 코드를 입력하세요", placeholder="예: YUALING2026")
            final_price = total_price
            if coupon == "YUALING2026":
                final_price = int(total_price * 0.9)
                st.success("🎉 10% 할인 쿠폰이 적용되었습니다!")
            elif coupon:
                st.warning("유효하지 않은 쿠폰 코드입니다.")

            st.markdown(f"### 💳 총 결제 금액: **₩{final_price:,}** (할인 전 ₩{total_price:,})")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🛍️ 주문하기 (가상 결제)"):
                    st.balloons()
                    st.success("성공적으로 주문이 완료되었습니다! 감사합니다 💜")
                    order_info = {
                        "items": list(st.session_state.cart),
                        "total": final_price
                    }
                    st.session_state.order_history.append(order_info)
                    st.session_state.cart = []
            with c2:
                if st.button("🗑️ 장바구니 비우기"):
                    st.session_state.cart = []
                    st.rerun()

    elif menu == "✨ AI 스타일 큐레이터":
        st.title("💜 YuaLing PyTorch CNN 스타일 큐레이터")
        st.markdown("옷 사진을 올리면 딥러닝 모델(`FashionCNN`)이 분석해서 코디 팁을 추천해 드려요! 👗")
        
        uploaded_file = st.file_uploader("패션 이미지를 업로드해 주세요 (PNG, JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="업로드한 아이템", use_container_width=True)
            with col2:
                st.markdown("### 🔍 모델 분석 결과")
                if st.button("✨ CNN 스타일 분석 시작하기"):
                    with st.spinner("딥러닝 모델 추론 중... ⏳"):
                        category, comment = predict_fashion_image(image)
                    st.session_state.last_result = {"category": category, "comment": comment}
                    st.success(f"**🏷️ 예측 카테고리:** {category}")
                    st.info(f"**💬 추천 코멘트:**\n\n{comment}")
                
                if 'last_result' in st.session_state:
                    if st.button("❤️ 이 코디 찜하기"):
                        item = st.session_state.last_result
                        if item not in st.session_state.wishlist:
                            st.session_state.wishlist.append(item)
                            st.balloons()
                            st.toast("찜목록에 추가되었습니다! 🛍️")
                        else:
                            st.warning("이미 찜한 코디예요!")

    elif menu == "📖 시즌 스타일 가이드":
        st.title("📖 YuaLing Trend Style Guide")
        st.markdown("유아링 샵의 인기 트렌드를 모아둔 **스타일 가이드** 공간입니다 ✨")
        st.markdown("- 🌸 **원피스룩:** 플로럴 패턴과 가디건 조합으로 청순함 극대화")
        st.markdown("- 👚 **블라우스룩:** 타이 디테일과 시스루로 우아한 무드 연출")
        st.markdown("- 🌊 **비키니/바캉스룩:** 홀터넥과 플로럴 패턴으로 핫한 휴양지 패션")
        st.markdown("- 👗 **스커트룩:** 미니 플리츠와 하이삭스로 발랄한 무드 연출")
        st.markdown("- 👖 **팬츠룩:** 루즈핏 데님과 크롭탑으로 힙한 스트릿 감성")

    elif menu == "👑 사장님 상품 관리":
        st.title("👑 YuaLing 대표님 상품 등록실")
        st.markdown("새로운 신상 아이템을 직접 등록해보세요 ✨")

        new_p_name = st.text_input("상품 이름")
        new_p_price = st.number_input("상품 가격 (원)", min_value=1000, step=1000, value=35000)
        new_p_cat = st.selectbox("카테고리 선택", ["원피스", "블라우스", "비키니/바캉스", "스커트", "팬츠/아우터"])
        new_p_desc = st.text_area("상품 설명")

        if st.button("🚀 신상 등록하기"):
            if new_p_name and new_p_desc:
                st.session_state.products_db[new_p_name] = {
                    "price": new_p_price,
                    "category": new_p_cat,
                    "rating": 5.0,
                    "desc": new_p_desc,
                    "reviews": [{"text": "신상 너무 기대돼요 대박나세요! 💜", "star": 5}]
                }
                st.success(f"'{new_p_name}' 상품이 성공적으로 등록되었습니다! 🎉")
            else:
                st.warning("상품 이름과 설명을 모두 입력해주세요.")

    else:
        st.title(f"👤 {st.session_state.user_name}님의 마이 페이지")
        st.markdown("내가 찜해둔 상품 및 지난 주문 내역을 모아보는 공간이에요 🛍️")

        tab1, tab2 = st.tabs(["❤️ 찜목록", "📦 주문 내역"])

        with tab1:
            if len(st.session_state.wishlist) == 0:
                st.info("아직 찜한 상품이나 코디가 없어요!")
            else:
                for idx, saved in enumerate(st.session_state.wishlist):
                    with st.expander(f"✨ 찜한 항목 #{idx+1} [{saved['category']}]"):
                        st.write(saved['comment'])
                if st.button("🗑️ 전체 찜목록 비우기"):
                    st.session_state.wishlist = []
                    st.rerun()

        with tab2:
            if len(st.session_state.order_history) == 0:
                st.info("아직 주문한 내역이 없어요!")
            else:
                for idx, order in enumerate(st.session_state.order_history):
                    st.markdown(f"### 📦 주문 내역 #{idx+1}")
                    for itm in order['items']:
                        st.markdown(f" - {itm['name']} (₩{itm['price']:,})")
                    st.markdown(f"**결제 총액:** ₩{order['total']:,}")
                    st.markdown("---")
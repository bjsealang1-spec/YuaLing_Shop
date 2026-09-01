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
    def __init__(self, num_classes=6):
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

class_names = ["원피스", "스커트", "팬츠", "블라우스", "비키니/바캉스", "스타킹/양말"]

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
# 2. 세션 상태 초기화
# ---------------------------------------------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_name' not in st.session_state:
    st.session_state.user_name = "여니"
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
if 'points' not in st.session_state:
    st.session_state.points = 3000
if 'checked_in_today' not in st.session_state:
    st.session_state.checked_in_today = False

if 'products_db' not in st.session_state:
    st.session_state.products_db = {
        # --- 원피스 라인 (15개) ---
        "라벤더 실크 원피스": {"price": 45000, "category": "원피스", "rating": 4.9, "desc": "하늘하늘한 실루엣과 고급스러운 라벤더 컬러가 매력적인 데이트룩 원피스", "reviews": [{"text": "핏이 정말 공주님 같아요! 💜", "star": 5}]},
        "로맨틱 프릴 캉캉 원피스": {"price": 49000, "category": "원피스", "rating": 5.0, "desc": "걸을 때마다 살랑거리는 프릴 디테일이 청순함을 극대화해주는 캉캉 원피스", "reviews": [{"text": "입고 여행 갔는데 인생샷 건졌어요 ✨", "star": 5}]},
        "플로럴 샤링 미니 원피스": {"price": 42000, "category": "원피스", "rating": 4.8, "desc": "바디라인을 예쁘게 잡아주는 샤링 디테일과 화사한 플로럴 패턴의 조합", "reviews": [{"text": "여리여리해 보이고 너무 예뻐요!", "star": 5}]},
        "진주 카라 스퀘어 원피스": {"price": 52000, "category": "원피스", "rating": 4.9, "desc": "단아하면서도 고급스러운 진주 버튼과 스퀘어 넥라인의 하객룩 원피스", "reviews": [{"text": "단정하고 고급스러워 보여요.", "star": 5}]},
        "체크 트위드 미니 원피스": {"price": 56000, "category": "원피스", "rating": 4.7, "desc": "클래식한 트위드 소재와 세련된 체크 패턴으로 페미닌한 무드 완성", "reviews": [{"text": "재질도 도톰하고 핏이 딱 잡혀요!", "star": 5}]},
        "모던 블랙 셔츠 원피스": {"price": 43000, "category": "원피스", "rating": 4.8, "desc": "툭 걸치기만 해도 시크한 매력이 터지는 벨트 세트 셔츠 원피스", "reviews": [{"text": "데일리로 입기 너무 편해요.", "star": 5}]},
        "여리여리 니트 롱 원피스": {"price": 48000, "category": "원피스", "rating": 4.9, "desc": "부드러운 촉감으로 몸을 감싸주는 가을/겨울 필수 청순 니트 원피스", "reviews": [{"text": "촉감 대박 부드러워요!", "star": 5}]},
        "새틴 홀터넥 롱 원피스": {"price": 54000, "category": "원피스", "rating": 5.0, "desc": "우아한 광택감으로 파티나 특별한 날 주인공으로 만들어주는 롱 원피스", "reviews": [{"text": "실물깡패입니다 꼭 사세요", "star": 5}]},
        "코튼 퍼프 미니 원피스": {"price": 39000, "category": "원피스", "rating": 4.6, "desc": "상큼 발랄한 분위기를 주는 탄탄한 코튼 소재의 퍼프 원피스", "reviews": [{"text": "상큼하고 귀여워요~", "star": 5}]},
        "시스루 레이스 슬립 원피스": {"price": 47000, "category": "원피스", "rating": 4.8, "desc": "레이어드해서 입기 좋은 로맨틱한 무드의 시스루 슬립 원피스", "reviews": [{"text": "레이어드 끝판왕!", "star": 5}]},
        "빈티지 도트 랩 원피스": {"price": 44000, "category": "원피스", "rating": 4.7, "desc": "체형 구애 없이 예쁘게 떨어지는 플레어라인 도트 랩 원피스", "reviews": [{"text": "날씬해 보여서 좋네요.", "star": 5}]},
        "벨벳 오프숄더 원피스": {"price": 59000, "category": "원피스", "rating": 4.9, "desc": "고급스러운 벨벳 소재와 오프숄더 넥라인의 연말 파티룩 원피스", "reviews": [{"text": "연말에 이거 하나면 끝!", "star": 5}]},
        "플리츠 카라 미디 원피스": {"price": 51000, "category": "원피스", "rating": 4.8, "desc": "단정하면서도 스커트 플리츠 포인트로 발랄함을 더한 미디 원피스", "reviews": [{"text": "출근룩으로 딱입니다.", "star": 5}]},
        "보헤미안 자수 롱 원피스": {"price": 53000, "category": "원피스", "rating": 4.6, "desc": "휴양지나 피크닉룩으로 제격인 감성 만점 자수 디테일 롱 원피스", "reviews": [{"text": "여신룩 등극이요 ㅋㅋㅋ", "star": 5}]},
        "투톤 데님 미니 원피스": {"price": 46000, "category": "원피스", "rating": 4.7, "desc": "캐주얼하면서도 힙한 스타일링을 완성해 주는 투톤 데님 원피스", "reviews": [{"text": "핏 진짜 트렌디해요.", "star": 5}]},

        # --- 블라우스 라인 (15개) ---
        "리본 퍼프 블라우스": {"price": 36000, "category": "블라우스", "rating": 4.7, "desc": "청순한 퍼프 소매와 리본 포인트가 사랑스러운 데일리 블라우스", "reviews": [{"text": "얼굴이 화사해 보여요 ✨", "star": 5}]},
        "시스루 타이 블라우스": {"price": 38000, "category": "블라우스", "rating": 4.9, "desc": "우아한 타이 리본 디테일과 은은한 시스루로 여리여리한 무드 연출", "reviews": [{"text": "슬랙스랑 입으니 진심 고급스러워요.", "star": 5}]},
        "새틴 브이넥 블라우스": {"price": 39000, "category": "블라우스", "rating": 4.8, "desc": "은은한 광택감이 도는 고급 새틴 소재의 세련된 오피스룩 블라우스", "reviews": [{"text": "촉감이 너무 부드럽고 예뻐요.", "star": 5}]},
        "레이스 넥 프릴 블라우스": {"price": 34000, "category": "블라우스", "rating": 4.6, "desc": "넥라인 레이스 프릴이 사랑스러운 빈티지 무드의 블라우스", "reviews": [{"text": "레이어드해서 입기 최고예요!", "star": 5}]},
        "스퀘어넥 핀턱 블라우스": {"price": 37000, "category": "블라우스", "rating": 4.8, "desc": "쇄골라인이 예뻐 보이는 스퀘어 넥과 정교한 핀턱 디테일 블라우스", "reviews": [{"text": "목선이 슬림해 보여요.", "star": 5}]},
        "시폰 셔링 셔츠 블라우스": {"price": 35000, "category": "블라우스", "rating": 4.7, "desc": "하늘하늘한 시폰 소재에 은은한 셔링으로 여성스러움을 강조한 블라우스", "reviews": [{"text": "하늘하늘해서 여리여리 핏 완성!", "star": 5}]},
        "진주 버튼 브이넥 블라우스": {"price": 41000, "category": "블라우스", "rating": 4.9, "desc": "영롱한 진주 버튼이 포인트가 되어주는 단아한 무드의 블라우스", "reviews": [{"text": "단추 디테일 미쳤어요 너무 예쁨", "star": 5}]},
        "시스루 도트 패턴 블라우스": {"price": 36000, "category": "블라우스", "rating": 4.7, "desc": "귀여운 도트 패턴과 시스루 원단의 조화로 사랑스러운 데이트 블라우스", "reviews": [{"text": "남친이 예쁘다고 난리네요 ㅋㅋ", "star": 5}]},
        "언발란스 랩 블라우스": {"price": 39000, "category": "블라우스", "rating": 4.8, "desc": "유니크한 랩 디자인으로 세련된 실루엣을 연출해 주는 블라우스", "reviews": [{"text": "흔하지 않은 디자인이라 좋아요.", "star": 5}]},
        "카라 스티치 데님 블라우스": {"price": 38000, "category": "블라우스", "rating": 4.6, "desc": "캐주얼한 데님 소재에 스티치 포인트로 힙한 감성을 더한 블라우스", "reviews": [{"text": "힙하면서도 여성스러워요.", "star": 5}]},
        "오간자 시스루 퍼프 블라우스": {"price": 42000, "category": "블라우스", "rating": 4.9, "desc": "볼륨감 있는 오간자 소매가 고급스러움을 극대화해주는 블라우스", "reviews": [{"text": "소매 핏이 살아있어요!", "star": 5}]},
        "플로럴 보타이 블라우스": {"price": 37000, "category": "블라우스", "rating": 4.7, "desc": "화사한 플로럴 패턴과 보타이 세트로 우아함을 살린 블라우스", "reviews": [{"text": "얼굴에 형광등 켠 줄 알았어요.", "star": 5}]},
        "코튼 베이직 셔츠 블라우스": {"price": 32000, "category": "블라우스", "rating": 4.5, "desc": "유행을 타지 않는 베이직한 디자인으로 어디에나 코디하기 좋은 셔츠", "reviews": [{"text": "기본템으로 최고입니다.", "star": 5}]},
        "벌룬 소매 브이넥 블라우스": {"price": 35000, "category": "블라우스", "rating": 4.8, "desc": "귀여운 볼륨 벌룬 소매로 미운 팔뚝살을 싹 가려주는 블라우스", "reviews": [{"text": "팔뚝 커버 완벽해요 감격 ㅠㅠ", "star": 5}]},
        "프릴 넥 세라 블라우스": {"price": 38000, "category": "블라우스", "rating": 4.7, "desc": "러블리한 세라 카라와 프릴 장식이 돋보이는 소녀 감성 블라우스", "reviews": [{"text": "교복처럼 자주 입게 돼요.", "star": 5}]},

        # --- 스커트 라인 (15개) ---
        "하이웨이스트 플리츠 스커트": {"price": 32000, "category": "스커트", "rating": 4.8, "desc": "다리가 길어 보이는 하이웨이스트 라인의 트렌디한 플리츠 스커트", "reviews": [{"text": "허리 밴딩이 편해서 자주 입어요~", "star": 5}]},
        "머메이드 롱 미디 스커트": {"price": 38000, "category": "스커트", "rating": 4.9, "desc": "곡선미를 예쁘게 살려주는 우아한 실루엣의 머메이드 스커트", "reviews": [{"text": "몸매가 엄청 예뻐 보여요 대박!", "star": 5}]},
        "체크 버튼 H라인 스커트": {"price": 33000, "category": "스커트", "rating": 4.7, "desc": "캐주얼하면서도 깔끔하게 떨어지는 핏의 H라인 스커트", "reviews": [{"text": "기장도 딱 좋고 날씬해 보여요.", "star": 5}]},
        "레더 핀턱 미니 스커트": {"price": 36000, "category": "스커트", "rating": 4.8, "desc": "시크하고 힙한 무드를 완성해 주는 부드러운 에코 레더 미니 스커트", "reviews": [{"text": "클럽이나 연말룩으로 짱이에요.", "star": 5}]},
        "언발란스 셔링 스커트": {"price": 35000, "category": "스커트", "rating": 4.7, "desc": "볼륨감 있는 셔링 디테일로 다리 라인을 예뻐 보이게 하는 스커트", "reviews": [{"text": "핏이 진짜 유니크하고 예뻐요.", "star": 5}]},
        "트위드 골드버튼 스커트": {"price": 41000, "category": "스커트", "rating": 4.9, "desc": "고급스러운 골드 버튼과 트위드 소재의 완벽한 조합 하이퀄리티 스커트", "reviews": [{"text": "세트로 산 자켓이랑 찰떡이에요.", "star": 5}]},
        "코튼 카고 롱 스커트": {"price": 37000, "category": "스커트", "rating": 4.6, "desc": "스트리트 감성이 물씬 풍기는 트렌디한 포켓 카고 롱 스커트", "reviews": [{"text": "활동하기 너무 편안해요.", "star": 5}]},
        "새틴 미디 플레어 스커트": {"price": 39000, "category": "스커트", "rating": 4.8, "desc": "걸을 때마다 우아한 광택감과 플레어 라인이 휘날리는 스커트", "reviews": [{"text": "여성스러움 끝판왕입니다.", "star": 5}]},
        "데님 앞트임 롱 스커트": {"price": 36000, "category": "스커트", "rating": 4.7, "desc": "부담 없는 앞트임 디테일로 활동성을 높인 베이직 데님 롱 스커트", "reviews": [{"text": "핏이 일자로 떨어져서 날씬해 보여요.", "star": 5}]},
        "체크 니트 밴딩 스커트": {"price": 34000, "category": "스커트", "rating": 4.6, "desc": "따뜻하고 포근한 니트 소재에 체크 패턴을 더한 윈터 밴딩 스커트", "reviews": [{"text": "겨울 내내 이것만 입을 듯!", "star": 5}]},
        "슬릿 미디 H라인 스커트": {"price": 33000, "category": "스커트", "rating": 4.8, "desc": "뒤트임 슬릿으로 편안함과 섹시함을 동시에 잡은 오피스 H스커트", "reviews": [{"text": "오피스룩으로 완벽합니다.", "star": 5}]},
        "플로럴 프릴 롱 스커트": {"price": 38000, "category": "스커트", "rating": 4.9, "desc": "하늘하늘한 쉬폰 소재에 플로럴 패턴이 가득 담긴 청순 롱 스커트", "reviews": [{"text": "바람 불 때 진짜 예뻐요 🌸", "star": 5}]},
        "모던 벨트 세트 스커트": {"price": 40000, "category": "스커트", "rating": 4.7, "desc": "세트로 구성된 깔끔한 벨트가 허리를 더욱 잘록하게 잡아주는 스커트", "reviews": [{"text": "벨트 세트라 구성이 혜자예요.", "star": 5}]},
        "테니스 베이직 미니 스커트": {"price": 29000, "category": "스커트", "rating": 4.7, "desc": "영하고 발랄한 영원한 스테디셀러 베이직 테니스 스커트", "reviews": [{"text": "가격도 착하고 핏도 굿!", "star": 5}]},
        "벨벳 캉캉 롱 스커트": {"price": 42000, "category": "스커트", "rating": 4.8, "desc": "빛에 따라 은은하게 빛나는 벨벳 소재의 유니크 캉캉 롱 스커트", "reviews": [{"text": "고급스러움이 줄줄 흐릅니다.", "star": 5}]},

        # --- 팬츠/아우터 라인 (15개) ---
        "시크 오버핏 블레이저": {"price": 58000, "category": "팬츠/아우터", "rating": 4.9, "desc": "툭 걸쳐도 세련된 무드가 살아나는 오버핏 데일리 블레이저", "reviews": [{"text": "오피스룩으로 완벽합니다!", "star": 5}]},
        "와이드 루즈핏 코튼 팬츠": {"price": 36000, "category": "팬츠/아우터", "rating": 4.8, "desc": "하루 종일 편안하게 입기 좋은 내추럴 핏 와이드 팬츠", "reviews": [{"text": "진짜 편해서 매일 입고 있어요.", "star": 5}]},
        "크롭 윈드브레이커 자켓": {"price": 49000, "category": "팬츠/아우터", "rating": 4.7, "desc": "간절기 환절기에 툭 걸치기 좋은 트렌디한 크롭 아우터", "reviews": [{"text": "디자인도 귀엽고 가벼워요!", "star": 5}]},
        "하이웨이스트 부츠컷 슬랙스": {"price": 42000, "category": "팬츠/아우터", "rating": 4.9, "desc": "다리가 마법처럼 길어 보이는 완벽한 핏의 부츠컷 슬랙스", "reviews": [{"text": "인생 슬랙스 찾았습니다 ㅠㅠ", "star": 5}]},
        "투웨이 크롭 후드 집업": {"price": 45000, "category": "팬츠/아우터", "rating": 4.8, "desc": "다양한 연출이 가능한 투웨이 지퍼와 숏한 기장감의 후드 집업", "reviews": [{"text": "꾸안꾸룩으로 최고예요.", "star": 5}]},
        "링클프리 핀턱 와이드 팬츠": {"price": 39000, "category": "팬츠/아우터", "rating": 4.8, "desc": "구김이 가지 않아 하루 종일 깔끔한 핏을 유지하는 핀턱 팬츠", "reviews": [{"text": "출근할 때 다림질 안 해도 돼서 꿀!", "star": 5}]},
        "클래식 트렌치 코트": {"price": 89000, "category": "팬츠/아우터", "rating": 5.0, "desc": "가을 감성의 정석, 탄탄한 소재와 클래식한 무드의 트렌치 코트", "reviews": [{"text": "백화점 구경 갈 필요 없음 대박", "star": 5}]},
        "스트레이트 연청 데님 팬츠": {"price": 38000, "category": "팬츠/아우터", "rating": 4.7, "desc": "사계절 내내 활용하기 좋은 청량한 컬러감의 일자 데님 팬츠", "reviews": [{"text": "핏이 일자로 떨어져서 다리 두꺼운 거 다 가려져요.", "star": 5}]},
        "퍼 자켓 크롭 코트": {"price": 72000, "category": "팬츠/아우터", "rating": 4.9, "desc": "부드럽고 럭셔리한 퍼 촉감으로 한겨울에도 따뜻한 크롭 아우터", "reviews": [{"text": "입었을 때 공주 된 기분이에요.", "star": 5}]},
        "조거 셋업 스웨트 팬츠": {"price": 34000, "category": "팬츠/아우터", "rating": 4.6, "desc": "집 앞이나 원룸텔에서 편하게 입기 좋은 포근한 조거 팬츠", "reviews": [{"text": "너무 편해서 홈웨어로 딱입니다.", "star": 5}]},
        "레더 라이더 자켓": {"price": 79000, "category": "팬츠/아우터", "rating": 4.9, "desc": "시크함의 상징, 라이딩 감성을 담아낸 탄탄한 에코 레더 자켓", "reviews": [{"text": "시크 폭발 핏 작살나요.", "star": 5}]},
        "카고 포켓 스트링 팬츠": {"price": 41000, "category": "팬츠/아우터", "rating": 4.7, "desc": "밑단 스트링으로 투웨이 연출이 가능한 힙한 스트리트 카고 팬츠", "reviews": [{"text": "댄서 느낌 나고 너무 예뻐요.", "star": 5}]},
        "베이직 가디건 니트 자켓": {"price": 46000, "category": "팬츠/아우터", "rating": 4.8, "desc": "금장 버튼 포인트로 단정하면서도 고급스러운 니트 자켓 가디건", "reviews": [{"text": "단추가 고급스러워서 비싸 보여요.", "star": 5}]},
        "부츠컷 흑청 데님 팬츠": {"price": 39000, "category": "팬츠/아우터", "rating": 4.7, "desc": "다크한 흑청 컬러로 슬림한 레그라인을 완성해 주는 부츠컷 팬츠", "reviews": [{"text": "흑청 컬러감 미쳤습니다.", "star": 5}]},
        "숏 패딩 점퍼": {"price": 68000, "category": "팬츠/아우터", "rating": 4.8, "desc": "가볍지만 따뜻하고 트렌디한 숏 기장감의 빵빵한 패딩 점퍼", "reviews": [{"text": "한겨울에도 이것만 입어요 따뜻해요.", "star": 5}]},

        # --- 비키니/바캉스 라인 (15개) ---
        "글램 홀터넥 비키니": {"price": 42000, "category": "비키니/바캉스", "rating": 4.9, "desc": "바디라인을 슬림하고 볼륨감 있게 잡아주는 세련된 홀터넥 비키니", "reviews": [{"text": "컬러감도 고급스럽고 핏 대박이에요 🌊", "star": 5}]},
        "레이스 프릴 비치 원피스": {"price": 46000, "category": "비키니/바캉스", "rating": 5.0, "desc": "휴양지에서 로맨틱한 분위기를 완성해 주는 시스루 비치 커버업", "reviews": [{"text": "바닷가에서 사진 정말 잘 나와요!", "star": 5}]},
        "오프숄더 프릴 비키니 셋업": {"price": 44000, "category": "비키니/바캉스", "rating": 4.8, "desc": "어깨 라인을 여리여리하게 보여주는 사랑스러운 오프숄더 수영복", "reviews": [{"text": "체형 커버되면서 너무 예뻐요.", "star": 5}]},
        "하이웨스트 로맨틱 모노키니": {"price": 48000, "category": "비키니/바캉스", "rating": 4.9, "desc": "노출이 부담스러운 언니들도 예쁘게 입을 수 있는 우아한 모노키니", "reviews": [{"text": "군살 싹 잡아줘서 인생 수영복 등극!", "star": 5}]},
        "시스루 비치 망사 가디건": {"price": 35000, "category": "비키니/바캉스", "rating": 4.7, "desc": "수영복 위에 무심하게 걸치기 좋은 여리여리 비치 망사 가디건", "reviews": [{"text": "햇빛 차단도 되고 실루엣 예뻐요.", "star": 5}]},
        "트로피컬 패턴 랩 원피스": {"price": 43000, "category": "비키니/바캉스", "rating": 4.8, "desc": "휴양지 분위기를 물씬 풍기는 화사한 트로피컬 무드 비치웨어", "reviews": [{"text": "동남아 여행 갈 때 필수품!", "star": 5}]},
        "스퀘어넥 하이레그 모노키니": {"price": 47000, "category": "비키니/바캉스", "rating": 4.9, "desc": "트렌디한 스퀘어 넥라인과 다리가 길어 보이는 하이레그 모노키니", "reviews": [{"text": "다리 엄청 길어 보여요 최고.", "star": 5}]},
        "프릴 스커트 비키니 3P": {"price": 51000, "category": "비키니/바캉스", "rating": 5.0, "desc": "비키니에 플레어 스커트가 세트로 구성되어 부담 없는 3피스 세트", "reviews": [{"text": "스커트가 있어서 노출 부담 없어요.", "star": 5}]},
        "체크 패턴 홀터넥 모노키니": {"price": 45000, "category": "비키니/바캉스", "rating": 4.7, "desc": "빈티지한 체크 패턴으로 하이틴 감성을 살린 귀여운 모노키니", "reviews": [{"text": "사진 찍으면 색감 엄청 예쁘게 나와요.", "star": 5}]},
        "비치 롱 로브 가운": {"price": 39000, "category": "비키니/바캉스", "rating": 4.8, "desc": "바람에 휘날릴 때 예술인 럭셔리 휴양지 비치 롱 로브 가운", "reviews": [{"text": "이거 입고 리조트 걸으면 모델 된 줄 ㅋㅋ", "star": 5}]},
        "니트 소재 비치 셋업": {"price": 49000, "category": "비키니/바캉스", "rating": 4.8, "desc": "그물망 니트탑과 팬츠 세트로 구성된 힙한 바캉스룩", "reviews": [{"text": "수영복 위에 세트로 입기 딱 좋아요.", "star": 5}]},
        "심플 베이직 비키니": {"price": 38000, "category": "비키니/바캉스", "rating": 4.6, "desc": "유행을 타지 않는 깔끔한 디자인으로 질리지 않는 베이직 비키니", "reviews": [{"text": "기본이라 오히려 손이 자주 가요.", "star": 5}]},
        "셔링 포인트 브이넥 모노키니": {"price": 46000, "category": "비키니/바캉스", "rating": 4.9, "desc": "가슴 라인 셔링으로 볼륨감을 더해주는 페미닌한 모노키니", "reviews": [{"text": "라인 정말 예쁘게 잡아줍니다.", "star": 5}]},
        "플로럴 오프숄더 비치 원피스": {"price": 42000, "category": "비키니/바캉스", "rating": 4.7, "desc": "어깨를 드러내어 청순 섹시미를 극대화해주는 플로럴 비치 원피스", "reviews": [{"text": "휴가철 여신룩 완성이에요.", "star": 5}]},
        "네온 컬러 포인트 비키니": {"price": 41000, "category": "비키니/바캉스", "rating": 4.7, "desc": "태닝한 피부와 완벽하게 어우러지는 핫한 네온 컬러 비키니", "reviews": [{"text": "시선 집중용으로 최고입니다!", "star": 5}]},

        # --- 스타킹/양말 라인 (25개 추가) ---
        "실크 투명 블랙 시스루 스타킹": {"price": 6000, "category": "스타킹/양말", "rating": 4.9, "desc": "다리 라인이 은은하게 비치며 매끈하게 보정해 주는 데일리 블랙 시스루 스타킹", "reviews": [{"text": "올도 잘 안 나가고 다리 엄청 예뻐 보여요 ✨", "star": 5}]},
        "각선미 압박 보정 블랙 스타킹": {"price": 9000, "category": "스타킹/양말", "rating": 5.0, "desc": "탱탱하게 군살을 꽉 잡아주어 슬림한 레그라인을 완성해 주는 고탄력 압박 스타킹", "reviews": [{"text": "신을 때 쫀쫀하고 다리가 붓지 않아요!", "star": 5}]},
        "꾸안꾸 무릎위 반스타킹 (블랙)": {"price": 5000, "category": "스타킹/양말", "rating": 4.8, "desc": "스커트나 교복룩에 매치하기 좋은 귀엽고 스포티한 무드의 블랙 반스타킹(니삭스)", "reviews": [{"text": "흘러내리지 않고 짱짱해서 좋아요.", "side": 5}]},
        "골지 면 니트 반스타킹 (아이보리)": {"price": 5500, "category": "스타킹/양말", "rating": 4.7, "desc": "따뜻하고 포근한 골지 짜임으로 가을겨울 감성 코디 필수템 니삭스", "reviews": [{"text": "메리제인 슈즈랑 신으면 대존예입니다.", "star": 5}]},
        "섹시 힙업 백시어리스 스타킹": {"price": 11000, "category": "스타킹/양말", "rating": 4.9, "desc": "뒤쪽 라인 포인트로 유니크하고 아찔한 분위기를 연출해 주는 백세임 스타킹", "reviews": [{"text": "뒤태 포인트가 진짜 매력 있어요.", "star": 5}]},
        "겨울용 피치 기모 덧댄 타이즈": {"price": 12000, "category": "스타킹/양말", "rating": 5.0, "desc": "겉보기엔 살색 시스루지만 안감은 포근한 기모로 한겨울에도 따뜻한 가짜 살색 타이즈", "reviews": [{"text": "겨울 구원투수템 ㅠㅠ 한겨울에도 원피스 가능!", "star": 5}]},
        "러블리 레이스 프릴 반스타킹": {"price": 6500, "category": "스타킹/양말", "rating": 4.8, "desc": "발목 윗부분에 사랑스러운 레이스 프릴이 더해진 로맨틱 무드 니삭스", "reviews": [{"text": "치마랑 신었을 때 소녀 감성 폭발해요.", "star": 5}]},
        "시크 심플 무지 발목 양말 (블랙)": {"price": 3000, "category": "스타킹/양말", "rating": 4.6, "desc": "어떤 스니커즈에나 찰떡같이 어울리는 베이직 블랙 숏 삭스", "reviews": [{"text": "가성비 좋고 쫀쫀해서 대량구매 각.", "star": 5}]},
        "파스텔 라벤더 컬러 발목 양말": {"price": 3500, "category": "스타킹/양말", "rating": 4.8, "desc": "유아링샵 시그니처 감성을 담은 여리여리 화사한 라벤더 컬러 양말", "reviews": [{"text": "색감이 너무 고와서 포인트로 딱이에요 💜", "star": 5}]},
        "망사 시스루 섹시 네트 스타킹": {"price": 7000, "category": "스타킹/양말", "rating": 4.7, "desc": "유니크한 스트리트 힙합룩이나 파티룩에 포인트 주기 좋은 네트 망사 스타킹", "reviews": [{"text": "힙한 느낌 내고 싶을 때 최고!", "star": 5}]},
        "도트 패턴 시스루 패션 타이즈": {"price": 8000, "category": "스타킹/양말", "rating": 4.9, "desc": "귀여운 도트 물방울 패턴이 콕콕 박혀 있어 심플한 원피스에 포인트 주기 좋은 타이즈", "reviews": [{"text": "다리 심심할 때 신어주면 완전 포인트 돼요.", "star": 5}]},
        "하이웨이스트 쫀쫀 판탈롱 스타킹 (5개 세트)": {"price": 15000, "category": "스타킹/양말", "rating": 4.9, "desc": "흘러내림 걱정없이 배까지 편안하게 감싸주는 가성비 갑 판탈롱 스타킹 묶음", "reviews": [{"text": "쟁여두고 신기 너무 편하고 좋아요.", "star": 5}]},
        "스트라이프 스포티 니삭스": {"price": 5500, "category": "스타킹/양말", "rating": 4.7, "desc": "두 줄 스트라이프 배색으로 하이틴 체육복이나 테니스룩에 어울리는 반스타킹", "reviews": [{"text": "발랄해 보이고 다리도 길어 보여요.", "star": 5}]},
        "커피색 내추럴 스킨 스타킹": {"price": 5000, "category": "스타킹/양말", "rating": 4.8, "desc": "내추럴하고 자연스러운 살구톤으로 정장이나 승무원 룩에 필수인 스킨 스타킹", "reviews": [{"text": "색상이 인위적이지 않고 자연스러워요.", "star": 5}]},
        "토탈 릴렉스 발가락 양말": {"price": 4000, "category": "스타킹/양말", "rating": 4.5, "desc": "발가락 사이사이 땀 흡수와 편안함을 극대화해 주는 기능성 토삭스", "reviews": [{"text": "엄청 편하고 쾌적합니다 ㅎㅎ", "star": 5}]},
        "벨벳 리본 포인트 블랙 반스타킹": {"price": 7500, "category": "스타킹/양말", "rating": 4.9, "desc": "종아리 옆쪽에 고급스러운 벨벳 리본이 달려있는 러블리 니삭스", "reviews": [{"text": "실물로 보면 리본이 진짜 고급져요.", "star": 5}]},
        "무릎 위 오버니삭스 (그레이)": {"price": 6000, "category": "스타킹/양말", "rating": 4.8, "desc": "다리를 포근하게 감싸주면서 슬림하게 만들어주는 그레이 컬러 오버니삭스", "reviews": [{"text": "따뜻하고 핏이 일자로 이쁘게 떨어져요.", "star": 5}]},
        "자외선 차단 쿨링 덧신 양말": {"price": 3500, "category": "스타킹/양말", "rating": 4.7, "desc": "로퍼나 스니커즈 신을 때 절대 벗겨지지 않는 실리콘 덧신", "reviews": [{"text": "진짜 절대 안 벗겨져서 인생 덧신이에요!", "star": 5}]},
        "체크 패턴 포인트 패션 미들삭스": {"price": 4500, "category": "스타킹/양말", "rating": 4.6, "desc": "클래식한 체크 패턴으로 로퍼나 워커에 신기 좋은 미들 삭스", "reviews": [{"text": "포인트 주기 은근히 좋아요.", "star": 5}]},
        "레오파트 호피 패턴 시스루 타이즈": {"price": 8500, "category": "스타킹/양말", "rating": 4.7, "desc": "과감하면서도 매혹적인 호피 패턴으로 스타일리시함을 뽐내는 타이즈", "reviews": [{"text": "유니크한 코디 할 때 대박입니다.", "star": 5}]},
        "발레코어 리본 스트랩 타이즈": {"price": 9500, "category": "스타킹/양말", "rating": 5.0, "desc": "발레리나 감성의 스트랩 리본 디테일이 살아있는 청순 발랄 시스루 타이즈", "reviews": [{"text": "요즘 트렌드 그 자체! 너무 예뻐요.", "star": 5}]},
        "논슬립 요가 필라테스 토삭스": {"price": 6000, "category": "스타킹/양말", "rating": 4.8, "desc": "바닥면에 미끄럼 방지 실리콘 처리가 되어 운동할 때 안전한 전용 양말", "reviews": [{"text": "필라테스 할 때 요거만 신어요.", "star": 5}]},
        "브이컷 스판 승무원 스타킹": {"price": 6000, "category": "스타킹/양말", "rating": 4.8, "desc": "오랫동안 서 있어도 다리가 편안하도록 설계된 고탄력 프리미엄 스타킹", "reviews": [{"text": "근무할 때 신기 아주 튼튼하고 좋아요.", "star": 5}]},
        "알록달록 스마일 자수 미들삭스": {"price": 4000, "category": "스타킹/양말", "rating": 4.7, "desc": "귀여운 스마일 자수가 콕 박혀 있어 기분까지 좋아지는 캐주얼 패션 양말", "reviews": [{"text": "신을 때마다 기분이 좋아져요 💛", "star": 5}]},
        "소프트 수면 털 양말 세트": {"price": 7000, "category": "스타킹/양말", "rating": 4.9, "desc": "원룸텔이나 집에서 잠잘 때 발을 따뜻하게 지켜주는 극세사 수면 양말", "reviews": [{"text": "수면 양말 이거 없으면 겨울 못 나요 극락 푹신함", "star": 5}]}
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
    input_name = st.sidebar.text_input("닉네임을 입력하세요", value="여니", key="sidebar_input_name")
    if st.sidebar.button("로그인하기", key="sidebar_login_btn"):
        st.session_state.logged_in = True
        st.session_state.user_name = input_name
        st.rerun()
else:
    st.sidebar.success(f"대표 **{st.session_state.user_name}**님 환영해요! 🎉")
    st.sidebar.info(f"💰 보유 적립금: **{st.session_state.points:,}원**")
    
    if not st.session_state.checked_in_today:
        if st.sidebar.button("📅 출석체크 (+1,000원)", key="sidebar_checkin_btn"):
            st.session_state.points += 1000
            st.session_state.checked_in_today = True
            st.success("출석체크 완료! 1,000원이 적립되었어요! 🎉")
            st.rerun()
    else:
        st.sidebar.caption("✅ 오늘의 출석체크 완료!")

    if st.sidebar.button("로그아웃", key="sidebar_logout_btn"):
        st.session_state.logged_in = False
        st.rerun()
    
    st.sidebar.markdown("---")
    if st.sidebar.button("👑 사장님 상품 등록실 열기", key="sidebar_admin_btn"):
        st.session_state.current_tab = "👑 상품등록"
        st.rerun()

# ---------------------------------------------------------
# 4. 상단 배너 및 탭바 UI
# ---------------------------------------------------------
if not st.session_state.logged_in:
    st.title("💜 YuaLing Shop에 오신 것을 환영합니다!")
    st.markdown("사이드바에서 **로그인**을 진행해 주세요! ✨")
else:
    st.markdown(f"### 🛍️ YuaLing Hot Menu (총 상품수: {len(st.session_state.products_db)}개)")
    q_cols = st.columns(5)
    with q_cols[0]:
        if st.button("🔥 베스트 랭킹", key="top_q_best"):
            st.session_state.current_tab = "🔥 베스트"
            st.session_state.selected_product = None
            st.rerun()
    with q_cols[1]:
        if st.button("🌸 전체상품", key="top_q_dress"):
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

            st.markdown("---")
            st.subheader("💬 생생 고객 리뷰")
            if p_info['reviews']:
                for rev in p_info['reviews']:
                    st.markdown(f"⭐️ {rev.get('star', 5)}점 | {rev['text']}")
            else:
                st.info("아직 작성된 리뷰가 없어요. 첫 리뷰의 주인공이 되어보세요! ✨")

            with st.form(key="review_form"):
                st.markdown("**✏️ 나도 리뷰 남기기**")
                user_star = st.slider("별점", 1, 5, 5, key="new_review_star")
                user_review_text = st.text_input("후기를 남겨주세요 (예: 핏이 너무 예뻐요!)", key="new_review_text")
                review_submit = st.form_submit_button("리뷰 등록하기")
                if review_submit and user_review_text:
                    p_info['reviews'].append({"text": user_review_text, "star": user_star})
                    st.success("소중한 리뷰가 등록되었습니다! 💜")
                    st.rerun()

        else:
            categories = ["전체", "원피스", "블라우스", "스커트", "팬츠/아우터", "비키니/바캉스", "스타킹/양말"]
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

            # --- [에이블리 감성 퀵 링크 섹션] ---
            st.markdown("---")
            st.markdown("### ⚡ YuaLing 퀵 바로가기 메뉴")
            
            quick_c1, quick_c2 = st.columns(2)
            with quick_c1:
                if st.button("📂 카테고리 전체 모아보기 >"):
                    st.session_state.current_tab = "🏠 홈"
                    st.rerun()
                if st.button("🔥 요즘코디 오버핏 추천 >"):
                    st.session_state.current_tab = "🔥 베스트"
                    st.rerun()
            with quick_c2:
                if st.button("💜 YuaLing 스토어 소개 >"):
                    st.toast("국내 최고 라벤더 감성 쇼핑몰 유아링샵입니다! ✨")
                if st.button("🛍️ 블리에 SALE 특별전 >"):
                    st.session_state.current_tab = "🔥 베스트"
                    st.rerun()

    elif tab == "🔥 베스트":
        st.title("🔥 YuaLing 실시간 베스트 랭킹")
        st.markdown("지금 가장 핫한 반응을 얻고 있는 인기 상품 TOP 5 야! ✨")
        
        sorted_products = sorted(st.session_state.products_db.items(), key=lambda x: x[1]['rating'], reverse=True)
        
        for rank, (item_name, info) in enumerate(sorted_products[:5], 1):
            st.markdown(f"### **{rank}위. {item_name}**")
            st.caption(f"카테고리: {info['category']} | 평점: ⭐️ {info['rating']} | 가격: ₩{info['price']:,}")
            st.write(f"{info['desc']}")
            if st.button(f"베스트 상품 구경하기 🔍", key=f"best_item_{rank}"):
                st.session_state.selected_product = item_name
                st.session_state.current_tab = "🏠 홈"
                st.rerun()
            st.markdown("---")

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
            
            use_points = st.checkbox(f"적립금 사용하기 (보유: {st.session_state.points:,}원)")
            if use_points:
                discount_from_points = min(st.session_state.points, final_price)
                final_price -= discount_from_points
                st.caption(f"✨ 적립금 {discount_from_points:,}원이 할인되었습니다!")

            st.markdown(f"### 💳 총 결제 금액: **₩{final_price:,}**")
            if st.button("🛍️ 주문하기", key="cart_order_btn"):
                if use_points:
                    st.session_state.points -= discount_from_points
                st.balloons()
                st.success("주문이 완료되었습니다! 💜")
                st.session_state.order_history.append({"items": list(st.session_state.cart), "total": final_price})
                st.session_state.cart = []

    elif tab == "👤 마이페이지":
        st.title(f"👤 {st.session_state.user_name}님의 마이페이지")
        st.info(f"💰 현재 보유 적립금: **{st.session_state.points:,}원**")
        
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
        new_p_cat = st.selectbox("카테고리", ["원피스", "블라우스", "비키니/바캉스", "스커트", "팬츠/아우터", "스타킹/양말"], key="reg_cat_select")
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
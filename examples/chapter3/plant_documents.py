"""多肉植物に関するサンプル文書。"""


from langchain_chroma import Chroma
from langchain_core.documents import Document

from llm_agent.config import create_openai_embeddings

SOURCE = "succulent-plants-doc"

PLANT_DOCUMENTS = [
    Document(
        page_content="""\
セダムはベンケイソウ科マンネングザ属で、日本にも自生しているポピュラーな多肉植物です。
種類が多くて葉の大きさや形状、カラーバリエーションも豊富なので、組み合わせて寄せ植えにしたり、庭のグランドカバーにしたりして楽しむことができます。
とても丈夫で育てやすく、多肉植物を初めて育てる方にもおすすめです。""",
        metadata={"source": SOURCE},
    ),
    Document(
        page_content="""\
熊童子はベンケイソウ科コチレドン属の多肉植物です。
葉に丸みや厚みがあり、先端には爪のような突起があることから「熊の手」という愛称で人気を集めています。
花はオレンジ色のベル型の花を咲かせることがあります。""",
        metadata={"source": SOURCE},
    ),
    Document(
        page_content="""\
エケベリアはベンケイソウ科エケベリア属の多肉植物で、メキシコなど中南米が原産です。
まるで花びらのように広がる肉厚な葉が特徴で、秋には紅葉も楽しめます。
品種が多く、室内でも気軽に育てられるので、人気のある多肉植物です。""",
        metadata={"source": SOURCE},
    ),
    Document(
        page_content="""\
ハオルチアは、春と秋に成長するロゼット形の多肉植物です。
密に重なった葉が放射状に展開し、幾何学的で整った株姿になるのが魅力です。
室内でも育てやすく手頃なサイズの多肉植物です。""",
        metadata={"source": SOURCE},
    ),
]

KUMATAROU_INFO = """\
熊童子はベンケイソウ科コチレドン属の多肉植物です。
葉に丸みや厚みがあり、先端には爪のような突起があることから「熊の手」という愛称で人気を集めています。
花はオレンジ色のベル型の花を咲かせることがあります。"""


def create_vectorstore() -> Chroma:
    """Chroma ベクトルストアを作成する。"""
    return Chroma.from_documents(PLANT_DOCUMENTS, embedding=create_openai_embeddings())

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from langchain-course!")
    information = """
    Stephen William Hawking (ur. 8 stycznia 1942 w Oksfordzie, zm. 14 marca 2018 w Cambridge[1]) – brytyjski fizyk teoretyczny i matematyczny.
    Stephen William Hawking był profesorem Uniwersytetu w Cambridge na katedrze Lucasa (1979–2009)[2][3] oraz w Kalifornijskim Instytucie Technicznym w Pasadenie, związanym też z Perimeter Institute for Theoretical Physics[4] (Waterloo, Ontario); członkiem Towarzystwa Królewskiego w Londynie (ang. Royal Society) i laureatem szeregu nagród, porównywalnych z Nagrodą Nobla jak Nagroda Alberta Einsteina (1978), Złoty Medal Królewskiego Towarzystwa Astronomicznego (1985), Nagroda Wolfa w dziedzinie fizyki (1988), Medal Copleya (2006) i Nagroda Specjalna Fizyki Fundamentalnej (2012).
    Specjalnościami Hawkinga były teoria względności i astrofizyka, w tym kosmologia. Badał Wielki Wybuch, czarne dziury, problem kwantowania grawitacji i jego związek z tymi zjawiskami. Za jego największe osiągnięcia uważa się[potrzebny przypis]:
    twierdzenie o osobliwościach – wspólnie z Rogerem Penrose’em udowodnił, że ogólna teoria względności (OTW) Einsteina zawiera osobliwości czasoprzestrzenne;
    przewidzenie, że czarne dziury powinny wysyłać promieniowanie, nazwane potem promieniowaniem Hawkinga[5][6][7] lub Bekensteina-Hawkinga; umożliwiło to obliczenie temperatury tych ciał i doprowadziło do paradoksu informacyjnego.
    Hawking był też wpływowym popularyzatorem; za życia opublikował kilka książek tego typu. Pierwsza z nich – Krótka historia czasu – znajdowała się na liście bestsellerów „British Sunday Times” przez rekordowy czas 237 tygodni (ponad czterech lat)[8]. W swojej twórczości popularyzatorskiej i publicznych wystąpieniach zabierał też głos na tematy inne niż fizyka i astronomia, takie jak astrobiologia, futurologia, filozofia – w tym metafizyka i etyka – oraz polityka. """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    llm = ChatOpenAI(temperature=0, model="gpt-5-nano")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()

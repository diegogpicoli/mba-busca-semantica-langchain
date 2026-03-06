import os

from dotenv import load_dotenv

from search import answer_question


def main() -> None:
    load_dotenv()
    k = int(os.getenv("TOP_K", "10"))

    print("Faca sua pergunta (digite 'sair' para encerrar):")

    while True:
        question = input("\nPERGUNTA: ").strip()
        if not question:
            continue

        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando chat.")
            break

        answer, _ = answer_question(question=question, k=k)
        print(f"RESPOSTA: {answer}")


if __name__ == "__main__":
    main()
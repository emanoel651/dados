import pandas as pd
from sqlalchemy import create_engine
import os

# --- CONFIGURAÇÕES ---
# 1. Coloque sua Connection String do Neon aqui
#    (Lembre-se de não subir este arquivo com a senha para repositórios públicos)
NEON_CONNECTION_STRING = "postgresql://neondb_owner:npg_syOZx7zMqag5@ep-wild-sun-acoe4vcq-pooler.sa-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

# 2. Caminho para o seu arquivo Excel/CSV
ARQUIVO_DE_DADOS = (
    "projeto_streamlit_kpi\\streamlit_app\\Plenum_e_Institulo_2024-2025_ordenado.xlsx"
)

# 3. Nome da tabela que será criada no banco de dados
NOME_DA_TABELA = "vendas"
# --------------------


def migrar_para_o_neon():
    """
    Lê um arquivo Excel/CSV e o insere em uma tabela no NeonDB.
    Se a tabela já existir, ela será substituída.
    """
    if not os.path.exists(ARQUIVO_DE_DADOS):
        print(f"Erro: Arquivo não encontrado em '{ARQUIVO_DE_DADOS}'")
        return

    print(f"Lendo dados de '{ARQUIVO_DE_DADOS}'...")
    try:
        if ARQUIVO_DE_DADOS.endswith(".csv"):
            df = pd.read_csv(ARQUIVO_DE_DADOS)
        else:
            df = pd.read_excel(ARQUIVO_DE_DADOS)

        print(f"{len(df)} linhas lidas com sucesso.")
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return

    try:
        print("Conectando ao banco de dados NeonDB...")
        engine = create_engine(NEON_CONNECTION_STRING)

        print(f"Enviando dados para a tabela '{NOME_DA_TABELA}'...")
        # Usamos to_sql para criar/substituir a tabela com os dados do DataFrame
        df.to_sql(
            NOME_DA_TABELA,
            con=engine,
            if_exists="replace",  # 'replace' apaga a tabela antiga e cria uma nova. Use 'append' para adicionar.
            index=False,  # Não salva o índice do pandas como uma coluna
        )
        print("\nDados migrados com sucesso para o NeonDB!")

    except Exception as e:
        print(f"Erro durante a conexão ou inserção no banco de dados: {e}")


if __name__ == "__main__":
    migrar_para_o_neon()

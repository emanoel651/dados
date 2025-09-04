import os
from sqlalchemy import create_engine, text

# --- CONFIGURAÇÕES ---
# 1. Cole aqui a sua string de conexão completa com o NeonDB.
#    (A mesma que você usa no secrets.toml)
NEON_CONNECTION_STRING = "postgresql://neondb_owner:npg_syOZx7zMqag5@ep-wild-sun-acoe4vcq-pooler.sa-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

# 2. Escreva o nome EXATO da sua coluna de Estado/UF, como está no banco de dados.
#    (Ex: "Estado", "UF", "estado_cliente", etc.)
NOME_DA_COLUNA_ESTADO = "Estado"

# 3. Nome da tabela que contém os dados.
NOME_DA_TABELA = "vendas"
# --------------------


def padronizar_coluna_estado():
    """
    Conecta ao banco de dados e executa um comando SQL para converter
    todos os valores da coluna de estado para letras maiúsculas.
    """
    if not all([NEON_CONNECTION_STRING, NOME_DA_COLUNA_ESTADO, NOME_DA_TABELA]):
        print(
            "Erro: Por favor, preencha todas as variáveis de configuração no topo do script."
        )
        return

    print("Conectando ao banco de dados...")
    try:
        engine = create_engine(NEON_CONNECTION_STRING)

        # O comando SQL para atualizar a coluna.
        # A função UPPER() do SQL converte o texto para maiúsculas.
        # As aspas duplas garantem que nomes de colunas com espaços ou caracteres especiais funcionem.
        sql_query = text(
            f'UPDATE "{NOME_DA_TABELA}" SET "{NOME_DA_COLUNA_ESTADO}" = UPPER("{NOME_DA_COLUNA_ESTADO}");'
        )

        with engine.connect() as connection:
            print(
                f"Executando atualização na coluna '{NOME_DA_COLUNA_ESTADO}' da tabela '{NOME_DA_TABELA}'..."
            )

            # Inicia uma transação
            trans = connection.begin()

            # Executa o comando
            connection.execute(sql_query)

            # Confirma (salva) a transação
            trans.commit()

            print(
                "\n✅ Sucesso! A coluna de estado foi padronizada para letras maiúsculas no banco de dados."
            )

    except Exception as e:
        print(f"\n❌ Erro durante a execução do script: {e}")
        print(
            "A operação foi revertida. Nenhuma alteração foi salva no banco de dados."
        )


if __name__ == "__main__":
    padronizar_coluna_estado()

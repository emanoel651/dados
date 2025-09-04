import os
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

# Carrega variáveis de ambiente de um arquivo .env (opcional, mas recomendado)
load_dotenv()

# --- CONFIGURAÇÃO ---
# 1. Caminho para a pasta onde estão seus PDFs.
#    O script assume que esta pasta está dentro do diretório do projeto.
PDF_DIRECTORY = Path("projeto_streamlit_kpi/projeto_streamlit_kpi/PDFS")

# 2. Pega a URL de conexão da variável de ambiente que você configurou.
DATABASE_URL = os.getenv("NEON_DATABASE_URL")

def upload_pdfs_to_db():
    """
    Conecta ao banco de dados, varre a pasta de PDFs e faz o upload
    de arquivos que ainda não existem na tabela 'arquivos_pdf'.
    """
    if not DATABASE_URL:
        print("❌ Erro: A variável de ambiente NEON_DATABASE_URL não foi definida.")
        return

    if not PDF_DIRECTORY.exists() or not PDF_DIRECTORY.is_dir():
        print(f"❌ Erro: O diretório de PDFs não foi encontrado em '{PDF_DIRECTORY.absolute()}'")
        return

    conn = None
    files_uploaded_count = 0
    files_skipped_count = 0

    try:
        # --- Conectar ao NeonDB ---
        print("🔌 Conectando ao banco de dados Neon...")
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        print("✅ Conexão bem-sucedida!")

        # --- Garantir que a tabela existe (opcional, mas seguro) ---
        cur.execute("""
            CREATE TABLE IF NOT EXISTS arquivos_pdf (
                id SERIAL PRIMARY KEY,
                nome_arquivo VARCHAR(255) NOT NULL UNIQUE,
                tipo_mime VARCHAR(100) NOT NULL,
                conteudo_pdf BYTEA NOT NULL,
                data_upload TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()

        # --- Varredura e Upload ---
        pdf_files = list(PDF_DIRECTORY.glob("*.pdf"))
        total_files = len(pdf_files)
        print(f"\n🔎 Encontrados {total_files} arquivos PDF na pasta.")

        for index, pdf_path in enumerate(pdf_files):
            file_name = pdf_path.name
            print(f"   [{index + 1}/{total_files}] Processando '{file_name}'...")

            # 1. Verificar se o arquivo já existe no banco
            cur.execute("SELECT id FROM arquivos_pdf WHERE nome_arquivo = %s;", (file_name,))
            if cur.fetchone():
                print("   ➡️ Arquivo já existe no banco de dados. Pulando.")
                files_skipped_count += 1
                continue

            # 2. Se não existir, ler e inserir
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()

                insert_sql = """
                    INSERT INTO arquivos_pdf (nome_arquivo, tipo_mime, conteudo_pdf)
                    VALUES (%s, %s, %s);
                """
                cur.execute(insert_sql, (file_name, 'application/pdf', pdf_bytes))
                print("   ⬆️  Arquivo inserido com sucesso!")
                files_uploaded_count += 1

            except Exception as e:
                print(f"   ❌ Erro ao ler ou inserir o arquivo '{file_name}': {e}")
                conn.rollback() # Desfaz a inserção com erro

        # --- Finalizar ---
        conn.commit() # Salva todas as inserções bem-sucedidas
        print("\n--- RESUMO ---")
        print(f"✔️ {files_uploaded_count} novo(s) arquivo(s) enviado(s) para o banco de dados.")
        print(f"⏭️ {files_skipped_count} arquivo(s) pulado(s) (já existiam).")
        print("🚀 Processo concluído!")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"\n❌ Erro de banco de dados: {error}")
    finally:
        if conn is not None:
            cur.close()
            conn.close()
            print("\n🔌 Conexão com o banco de dados fechada.")

if __name__ == "__main__":
    upload_pdfs_to_db()
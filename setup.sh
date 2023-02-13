mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"saurabhnair2391999@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
Footer

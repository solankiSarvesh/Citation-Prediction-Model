import streamlit as st
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import json
import os
import binascii

# Page configuration
st.set_page_config(
    page_title="ECIES Encryption Dashboard",
    page_icon="🔐",
    layout="wide"
)

# Custom CSS for attractive styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .st-emotion-cache-16idsys p {
        font-size: 1.1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #667eea;
    }
    .success-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d4edda;
        border: 2px solid #28a745;
        margin: 10px 0;
    }
    .info-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #d1ecf1;
        border: 2px solid #17a2b8;
        margin: 10px 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 25px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for keys
if 'priv_key' not in st.session_state:
    st.session_state.priv_key = ec.generate_private_key(ec.SECP256R1())
    st.session_state.pub_key = st.session_state.priv_key.public_key()

# ----------- HKDF Key Derivation -------------
def derive_key(shared_secret: bytes) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"ec-elgamal-hybrid",
    )
    return hkdf.derive(shared_secret)

# ----------- Encryption Function -------------
def encrypt_message(plaintext: str, pub_key):
    plaintext_bytes = plaintext.encode("utf-8")
    
    # Ephemeral key
    eph_priv = ec.generate_private_key(ec.SECP256R1())
    eph_pub = eph_priv.public_key()
    
    # Shared secret
    shared = eph_priv.exchange(ec.ECDH(), pub_key)
    
    # Symmetric key
    key = derive_key(shared)
    aes = AESGCM(key)
    iv = os.urandom(12)
    ct = aes.encrypt(iv, plaintext_bytes, None)
    
    # Ephemeral pubkey in uncompressed point format
    R_bytes = eph_pub.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    
    cipher_obj = {
        "curve": "secp256r1",
        "R": binascii.hexlify(R_bytes).decode(),
        "IV": binascii.hexlify(iv).decode(),
        "CT": binascii.hexlify(ct).decode(),
    }
    
    return cipher_obj

# ----------- Decryption Function -------------
def decrypt_message(cipher_json: dict, priv_key):
    try:
        R = binascii.unhexlify(cipher_json["R"])
        iv = binascii.unhexlify(cipher_json["IV"])
        ct = binascii.unhexlify(cipher_json["CT"])
        
        eph_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), R)
        
        # shared secret = d * R
        shared = priv_key.exchange(ec.ECDH(), eph_pub)
        key = derive_key(shared)
        aes = AESGCM(key)
        pt = aes.decrypt(iv, ct, None)
        
        return pt.decode("utf-8")
    except Exception as e:
        return f"Error: {str(e)}"

# Header
st.markdown("<h1 style='text-align: center; color: white; font-size: 3.5rem;'>🔐 ECIES Encryption System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #e0e0e0;'>Elliptic Curve Integrated Encryption Scheme</h3>", unsafe_allow_html=True)
st.markdown("---")

# Key Status Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class='info-box'>
        <h3 style='text-align: center; margin:0;'>🔑 Encryption Keys Active</h3>
        <p style='text-align: center; margin:5px 0 0 0;'>Using SECP256R1 Elliptic Curve</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Regenerate Keys", use_container_width=True):
        st.session_state.priv_key = ec.generate_private_key(ec.SECP256R1())
        st.session_state.pub_key = st.session_state.priv_key.public_key()
        st.success("✅ New keys generated successfully!")
        st.rerun()

st.markdown("---")

# Info Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h4 style='color: #667eea; margin-top: 0;'>🔗 HKDF Key Derivation</h4>
        <p style='color: #555; font-size: 0.9rem;'>SHA-256 based key derivation for symmetric encryption</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h4 style='color: #667eea; margin-top: 0;'>🔒 AES-GCM 256</h4>
        <p style='color: #555; font-size: 0.9rem;'>Authenticated encryption with associated data</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h4 style='color: #667eea; margin-top: 0;'>🛡️ ECDH Exchange</h4>
        <p style='color: #555; font-size: 0.9rem;'>Ephemeral keys for perfect forward secrecy</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Tabs
tab1, tab2 = st.tabs(["🔒 Encrypt", "🔓 Decrypt"])

# Encryption Tab
with tab1:
    st.markdown("### Encrypt Your Message")
    
    plaintext_input = st.text_area(
        "Enter plaintext message:",
        value="Hello Ma'am",
        height=150,
        key="encrypt_input"
    )
    
    if st.button("🔐 Encrypt Message", key="encrypt_btn", use_container_width=True):
        if plaintext_input:
            with st.spinner("Encrypting..."):
                cipher = encrypt_message(plaintext_input, st.session_state.pub_key)
                
            st.markdown("<div class='success-box'>", unsafe_allow_html=True)
            st.markdown("### ✅ Encryption Successful!")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("#### 📦 Encrypted Output (JSON):")
            st.json(cipher)
            
            # Copy button
            st.code(json.dumps(cipher, indent=2), language="json")
            
            st.info("💡 Copy the JSON above to decrypt it in the Decrypt tab!")
        else:
            st.error("⚠️ Please enter a message to encrypt!")

# Decryption Tab
with tab2:
    st.markdown("### Decrypt Your Message")
    
    # Example cipher for testing
    example_cipher = {
        "curve": "secp256r1",
        "R": "049575e2aea2a12c8da85a2a0a21f11f707557cd607f92359b260f2f1f48ac262bd803fbdeb8586a7dd0836ec52eb29baee7e440c02faf9993318325e00e43ee9e",
        "IV": "4528d80e73805c51f450e135",
        "CT": "6a2549a6632ced63397fd85e6b7f8300450d9e66535eef5ef41a28"
    }
    
    if st.button("📋 Load Example Cipher", key="load_example"):
        st.session_state.decrypt_input = json.dumps(example_cipher, indent=2)
    
    decrypt_input = st.text_area(
        "Paste encrypted JSON here:",
        value=st.session_state.get('decrypt_input', ''),
        height=200,
        key="decrypt_text"
    )
    
    if st.button("🔓 Decrypt Message", key="decrypt_btn", use_container_width=True):
        if decrypt_input:
            try:
                cipher_json = json.loads(decrypt_input)
                
                with st.spinner("Decrypting..."):
                    decrypted_text = decrypt_message(cipher_json, st.session_state.priv_key)
                
                if not decrypted_text.startswith("Error"):
                    st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                    st.markdown("### ✅ Decryption Successful!")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("#### 📄 Decrypted Message:")
                    st.markdown(f"<h2 style='color: white; background: #28a745; padding: 20px; border-radius: 10px; text-align: center;'>{decrypted_text}</h2>", unsafe_allow_html=True)
                else:
                    st.error(f"❌ {decrypted_text}")
                    
            except json.JSONDecodeError:
                st.error("⚠️ Invalid JSON format! Please enter valid JSON.")
            except Exception as e:
                st.error(f"❌ Decryption failed: {str(e)}")
        else:
            st.error("⚠️ Please enter cipher text to decrypt!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <p style='font-size: 1.1rem;'>🔐 Secure encryption using SECP256R1 elliptic curve cryptography</p>
    <p style='font-size: 0.9rem; opacity: 0.8;'>Perfect Forward Secrecy • AES-256-GCM • ECDH Key Exchange</p>
</div>
""", unsafe_allow_html=True)
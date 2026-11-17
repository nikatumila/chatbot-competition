import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

st.set_page_config(page_title="Simulasi Persaingan Chatbot AI", layout="wide")

st.title("Pemodelan Dinamika Sistem Persaingan Pasar Chatbot AI di Indonesia")
st.caption("Adaptasi model kompetisi (Shiflet, Modul 4.1): ChatGPT, Perplexity, Gemini, dan lainnya "
           "(Claude, Copilot, DeepSeek) berebut perhatian pengguna yang sama")

# ---------------------------------------------------------------
# 1. Data StatCounter (pangsa pasar chatbot AI Indonesia, %)
# ---------------------------------------------------------------
BULAN = ["Apr 25", "Mei 25", "Jun 25", "Jul 25", "Agu 25", "Sep 25", "Okt 25", "Nov 25", "Des 25", "Jan 26",
         "Feb 26", "Mar 26", "Apr 26", "Mei 26", "Jun 26", "Jul 26", "Agu 26", "Sep 26", "Okt 26", "Nov 26"]
DATA = {  # ChatGPT, Perplexity, Gemini, Claude, Copilot, DeepSeek
    "ChatGPT":    [76.31, 74.85, 74.83, 81.11, 80.60, 79.31, 79.69, 81.60, 81.28, 84.70, 85.00, 85.49, 84.19, 83.53, 81.11, 85.40, 86.13, 84.98, 59.36, 2.35],
    "Perplexity": [21.81, 22.46, 21.04, 15.02, 15.03, 16.41, 15.53, 14.32, 13.99, 10.13, 9.90, 8.40, 9.21, 10.31, 11.21, 7.75, 6.23, 8.00, 19.72, 46.89],
    "Gemini":     [1.11, 0.96, 1.44, 1.49, 1.78, 2.05, 2.66, 2.13, 2.71, 3.60, 3.26, 3.54, 4.34, 3.28, 4.23, 4.38, 5.47, 6.14, 19.19, 47.76],
    "Claude":     [0.43, 0.57, 1.68, 1.32, 1.54, 1.43, 1.44, 1.36, 1.36, 1.12, 1.38, 2.17, 1.66, 2.26, 2.87, 2.05, 1.83, 0.60, 1.42, 2.96],
    "Copilot":    [0.02, 0.88, 0.78, 0.83, 0.84, 0.78, 0.68, 0.59, 0.66, 0.45, 0.45, 0.40, 0.60, 0.63, 0.57, 0.41, 0.34, 0.28, 0.30, 0.04],
    "DeepSeek":   [0.32, 0.28, 0.24, 0.22, 0.21, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
}
NAMA = ["ChatGPT", "Perplexity", "Gemini", "Lainnya"]
WARNA = ["#10A37F", "#7B2D8E", "#4285F4", "#D97757"]
x_data = np.array([DATA["ChatGPT"], DATA["Perplexity"], DATA["Gemini"],
                   np.add(np.add(DATA["Claude"], DATA["Copilot"]), DATA["DeepSeek"])]).T
x_data = x_data / x_data.sum(axis=1, keepdims=True) * 100      # 4 stok, jumlah = 100
N_FIT = 18                                                       # Apr 2025 – Sep 2026 (Okt–Nov 2026 anomali)
dt = 0.01


# ---------------------------------------------------------------
# 2. Model: setiap aliran = k * penerima * pemberi / 100
# ---------------------------------------------------------------
def turunan(x, k_CP, k_GC, k_OC):
    c, p, g, o = x
    pindah_P_ke_C = k_CP * p * c / 100
    pindah_C_ke_G = k_GC * c * g / 100
    pindah_C_ke_O = k_OC * c * o / 100
    return np.array([pindah_P_ke_C - pindah_C_ke_G - pindah_C_ke_O,   # d(chatgpt)/dt
                     -pindah_P_ke_C,                                   # d(perplexity)/dt
                     pindah_C_ke_G,                                    # d(gemini)/dt
                     pindah_C_ke_O])                                   # d(lainnya)/dt


def simulasi(x0, k, bulan, ganti=None, langkah=dt):
    """Metode Euler. ganti = (t_mulai, k_baru) untuk skenario intervensi."""
    n = int(round(bulan / langkah))
    t = np.arange(n + 1) * langkah                # waktu berbasis indeks integer
    x = np.zeros((n + 1, 4))
    x[0] = x0
    for i in range(n):
        kk = ganti[1] if (ganti is not None and t[i] >= ganti[0]) else k
        x[i + 1] = x[i] + turunan(x[i], *kk) * langkah
    return t, x


@st.cache_data
def estimasi():
    """Kuadrat terkecil (k >= 0) untuk k_CP, k_GC dan pangsa awal pada 18 bulan pertama."""
    t = np.arange(N_FIT)

    def residu(z):
        x0 = np.append(z[2:5], 100 - z[2:5].sum())
        sol = solve_ivp(lambda tt, xx: turunan(xx, z[0], z[1], 0), (0, N_FIT - 1), x0, t_eval=t, rtol=1e-9, atol=1e-11)
        return (sol.y.T - x_data[:N_FIT]).ravel()

    z0 = np.r_[0.05, 0.05, x_data[0, :3]]
    r = least_squares(residu, z0, bounds=(np.zeros(5), np.r_[5, 5, 100, 100, 100]))
    return r.x[0], r.x[1], np.append(r.x[2:5], 100 - r.x[2:5].sum())


k_CP_fit, k_GC_fit, x0 = estimasi()

# ---------------------------------------------------------------
# 3. Parameter (sidebar)
# ---------------------------------------------------------------
st.sidebar.header("Parameter Model")
st.sidebar.caption(f"Nilai awal = hasil estimasi: k_CP = {k_CP_fit:.4f}, k_GC = {k_GC_fit:.4f}")
k_CP = st.sidebar.slider("k_CP: ChatGPT menarik pengguna Perplexity", 0.0, 0.30, float(round(k_CP_fit, 4)), 0.005, format="%.4f")
k_GC = st.sidebar.slider("k_GC: Gemini menarik pengguna ChatGPT", 0.0, 0.30, float(round(k_GC_fit, 4)), 0.005, format="%.4f")
k_OC = st.sidebar.slider("k_OC: Lainnya menarik pengguna ChatGPT", 0.0, 0.30, 0.0, 0.005, format="%.4f")
t_max = st.sidebar.slider("Durasi simulasi (bulan, t = 0 April 2025)", 24, 72, 72, 6)

st.sidebar.header("Skenario Intervensi (mulai Okt 2026)")
SKENARIO = {"Tidak ada": None,
            "A: Dorongan ekosistem Gemini (k_GC × 2)": lambda a, b, c: (a, 2 * b, c),
            "B: Retensi ChatGPT (k_GC × 0,5)": lambda a, b, c: (a, 0.5 * b, c),
            "C: Perplexity bangkit (k_CP dibalik)": lambda a, b, c: (-a, b, c),
            "D: Penantang tumbuh (k_OC = k_GC)": lambda a, b, c: (a, b, b)}
pilih = st.sidebar.selectbox("Pilih skenario", list(SKENARIO))

k = (k_CP, k_GC, k_OC)
t, x = simulasi(x0, k, t_max)
ganti = (18, SKENARIO[pilih](*k)) if SKENARIO[pilih] else None
t_s, x_s = simulasi(x0, k, t_max, ganti)


def bulan_ke(t_):
    return f"Bulan ke-{t_:.1f}" if t_ is not None else "Tidak tercapai"


def kalender(t_):
    if t_ is None:
        return ""
    m = int(t_) + 3
    return ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"][m % 12] + f" {2025 + m // 12}"


def menyalip(xx, tt):
    i = np.where(xx[:, 2] >= xx[:, 0])[0]
    return tt[i[0]] if len(i) else None


tab1, tab2, tab3, tab4 = st.tabs(["Data StatCounter", "Model vs Data", "Proyeksi & Skenario", "Verifikasi"])

# ---------------------------------------------------------------
# Tab 1: Data
# ---------------------------------------------------------------
with tab1:
    st.markdown("Pangsa pasar chatbot AI bulanan di Indonesia dari **StatCounter Global Stats**, April 2025 – "
                "November 2026 (pangsa rujukan kunjungan ke situs web). Okt–Nov 2026 adalah anomali dan "
                "tidak dipakai untuk fitting; Nov 2026 belum lengkap.")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axvspan(17.5, 19.5, color="orange", alpha=0.15, label="Anomali (Okt–Nov 2026)")
    for nama, warna in zip(DATA, WARNA[:3] + ["#D97757", "#E0B000", "#4D6BFE"]):
        ax.plot(DATA[nama], "-o", ms=3.5, color=warna, label=nama)
    ax.set_xticks(range(len(BULAN)))
    ax.set_xticklabels(BULAN, rotation=60, fontsize=8)
    ax.set_ylabel("Pangsa pasar (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(ncol=4, fontsize=8, loc="center left")
    st.pyplot(fig)
    st.dataframe({"Bulan": BULAN, **DATA})

# ---------------------------------------------------------------
# Tab 2: Model vs Data
# ---------------------------------------------------------------
with tab2:
    i_obs = [int(round(m / dt)) for m in range(N_FIT)]
    mae = np.abs(x[i_obs] - x_data[:N_FIT]).mean(axis=0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("k_CP hasil estimasi", f"{k_CP_fit:.4f} / bulan")
    c2.metric("k_GC hasil estimasi", f"{k_GC_fit:.4f} / bulan")
    c3.metric("MAE ChatGPT / Perplexity", f"{mae[0]:.2f} / {mae[1]:.2f} poin")
    c4.metric("MAE Gemini / Lainnya", f"{mae[2]:.2f} / {mae[3]:.2f} poin")

    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    ax2.axvspan(17.5, 19.5, color="orange", alpha=0.15, label="Tidak dipakai untuk fitting")
    batas = t <= 19
    for i in range(4):
        ax2.plot(t[batas], x[batas, i], color=WARNA[i], linewidth=2, label=f"{NAMA[i]} – model")
        ax2.plot(range(len(BULAN)), x_data[:, i], "o", color=WARNA[i], ms=5, label=f"{NAMA[i]} – data")
    ax2.set_xticks(range(len(BULAN)))
    ax2.set_xticklabels(BULAN, rotation=60, fontsize=8)
    ax2.set_ylabel("Pangsa pasar (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Model Kompetisi vs Data StatCounter", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(ncol=3, fontsize=8, loc="center left")
    st.pyplot(fig2)

# ---------------------------------------------------------------
# Tab 3: Proyeksi dan skenario
# ---------------------------------------------------------------
with tab3:
    i_puncak = int(np.argmax(x[:, 0]))
    t_salip, t_salip_s = menyalip(x, t), menyalip(x_s, t_s)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Puncak ChatGPT", f"{x[i_puncak, 0]:.1f} %", f"{bulan_ke(t[i_puncak])} ({kalender(t[i_puncak])})", delta_color="off")
    d2.metric("Gemini menyalip ChatGPT (dasar)", bulan_ke(t_salip), kalender(t_salip), delta_color="off")
    d3.metric("Gemini menyalip ChatGPT (skenario)", bulan_ke(t_salip_s), kalender(t_salip_s), delta_color="off")
    d4.metric(f"Pangsa akhir t={t_max} (ChatGPT / Gemini)", f"{x_s[-1, 0]:.1f} % / {x_s[-1, 2]:.1f} %")

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    for i in range(4):
        ax3.plot(t, x[:, i], color=WARNA[i], linewidth=2, label=f"{NAMA[i]} (dasar)")
        if ganti:
            ax3.plot(t_s, x_s[:, i], color=WARNA[i], linewidth=2, linestyle="--", label=f"{NAMA[i]} (skenario)")
    ax3.axvline(17, color="red", linestyle=":", label="Bulan terakhir fitting (Sep 2026)")
    if t_salip is not None:
        ax3.axvline(t_salip, color="gray", linestyle="-.", alpha=0.7, label=f"Gemini menyalip ChatGPT (bulan ke-{t_salip:.1f})")
    ax3.set_title("Proyeksi Pangsa Pasar Chatbot AI", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Waktu t (Bulan, t = 0 April 2025)")
    ax3.set_ylabel("Pangsa pasar (%)")
    ax3.set_ylim(0, 100)
    ax3.set_xlim(0, t_max)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(ncol=2, fontsize=8, loc="center left")
    st.pyplot(fig3)
    st.info("Proyeksi ini adalah skenario jika parameter tetap (what-if), bukan ramalan pasar.")

# ---------------------------------------------------------------
# Tab 4: Verifikasi
# ---------------------------------------------------------------
with tab4:
    rk = solve_ivp(lambda tt, xx: turunan(xx, *k), (0, t_max), x0, t_eval=t[::100], rtol=1e-10, atol=1e-12).y.T
    _, x_half = simulasi(x0, k, t_max, langkah=dt / 2)
    e1, e2, e3 = st.columns(3)
    e1.metric("Simpangan total dari 100%", f"{np.abs(x.sum(axis=1) - 100).max():.1e} poin")
    e2.metric("Selisih Euler vs Runge–Kutta", f"{np.abs(x[::100] - rk).max():.3f} poin")
    e3.metric("Selisih jika Δt dibagi dua", f"{np.abs(x_half[::200] - x[::100]).max():.3f} poin")
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    for i in range(4):
        ax4.plot(t[::100], rk[:, i], color="black", linewidth=1.5)
        ax4.plot(t, x[:, i], color=WARNA[i], linestyle="--", linewidth=2, label=f"{NAMA[i]} – Euler (dt = {dt})")
    ax4.plot([], [], color="black", label="Runge–Kutta (solve_ivp)")
    ax4.set_xlabel("Waktu t (Bulan)")
    ax4.set_ylabel("Pangsa pasar (%)")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(fontsize=8, loc="center left")
    st.pyplot(fig4)

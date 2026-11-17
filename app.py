import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

st.set_page_config(page_title="Simulasi Persaingan Chatbot AI", layout="wide")
st.title("Pemodelan Dinamika Sistem Persaingan Pasar Chatbot AI di Indonesia")
st.caption("Adaptasi model kompetisi Shiflet (Modul 4.1) untuk ChatGPT, Perplexity, Gemini, "
           "dan chatbot lainnya (Claude, Copilot, DeepSeek)")

# ---------------------------------------------------------------
# 1. Data StatCounter: pangsa pasar chatbot AI Indonesia (%), Apr 2025 - Nov 2026
# ---------------------------------------------------------------
BULAN = [f"{b} {y}" for y, bs in [(2025, "Apr Mei Jun Jul Agu Sep Okt Nov Des"),
                                  (2026, "Jan Feb Mar Apr Mei Jun Jul Agu Sep Okt Nov")] for b in bs.split()]
DATA = {
    "ChatGPT":    [76.31, 74.85, 74.83, 81.11, 80.60, 79.31, 79.69, 81.60, 81.28, 84.70, 85.00, 85.49, 84.19, 83.53, 81.11, 85.40, 86.13, 84.98, 59.36, 2.35],
    "Perplexity": [21.81, 22.46, 21.04, 15.02, 15.03, 16.41, 15.53, 14.32, 13.99, 10.13, 9.90, 8.40, 9.21, 10.31, 11.21, 7.75, 6.23, 8.00, 19.72, 46.89],
    "Gemini":     [1.11, 0.96, 1.44, 1.49, 1.78, 2.05, 2.66, 2.13, 2.71, 3.60, 3.26, 3.54, 4.34, 3.28, 4.23, 4.38, 5.47, 6.14, 19.19, 47.76],
    "Claude":     [0.43, 0.57, 1.68, 1.32, 1.54, 1.43, 1.44, 1.36, 1.36, 1.12, 1.38, 2.17, 1.66, 2.26, 2.87, 2.05, 1.83, 0.60, 1.42, 2.96],
    "Copilot":    [0.02, 0.88, 0.78, 0.83, 0.84, 0.78, 0.68, 0.59, 0.66, 0.45, 0.45, 0.40, 0.60, 0.63, 0.57, 0.41, 0.34, 0.28, 0.30, 0.04],
    "DeepSeek":   [0.32, 0.28, 0.24, 0.22, 0.21, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
}
NAMA = ["ChatGPT", "Perplexity", "Gemini", "Lainnya"]
WARNA = ["#10A37F", "#7B2D8E", "#4285F4", "#D97757", "#E0B000", "#4D6BFE"]
d = np.array(list(DATA.values()))
x_data = np.vstack([d[:3], d[3:].sum(0)]).T
x_data = x_data / x_data.sum(1, keepdims=True) * 100   # 4 stok, jumlah = 100
N_FIT, dt = 18, 0.01                                   # estimasi Apr 2025 - Sep 2026, langkah Euler (bulan)


# ---------------------------------------------------------------
# 2. Model: setiap aliran = k * penerima * pemberi / 100
# ---------------------------------------------------------------
def turunan(x, k_CP, k_GC, k_OC):
    c, p, g, o = x
    P_ke_C, C_ke_G, C_ke_O = k_CP * p * c / 100, k_GC * c * g / 100, k_OC * c * o / 100
    return np.array([P_ke_C - C_ke_G - C_ke_O, -P_ke_C, C_ke_G, C_ke_O])


def simulasi(x0, k, bulan, ganti=None, h=dt):
    """Metode Euler. ganti = (t_mulai, k_baru) untuk skenario intervensi."""
    t = np.arange(int(round(bulan / h)) + 1) * h
    x = np.zeros((len(t), 4))
    x[0] = x0
    for i in range(len(t) - 1):
        kk = ganti[1] if ganti and t[i] >= ganti[0] else k
        x[i + 1] = x[i] + turunan(x[i], *kk) * h
    return t, x


@st.cache_data
def estimasi():
    """Kuadrat terkecil (k >= 0) untuk k_CP, k_GC, dan pangsa awal."""
    def residu(z):
        x0 = np.append(z[2:], 100 - z[2:].sum())
        sol = solve_ivp(lambda _, xx: turunan(xx, z[0], z[1], 0), (0, N_FIT - 1), x0,
                        t_eval=np.arange(N_FIT), rtol=1e-9, atol=1e-11)
        return (sol.y.T - x_data[:N_FIT]).ravel()
    z = least_squares(residu, np.r_[0.05, 0.05, x_data[0, :3]], bounds=(0, [5, 5, 100, 100, 100])).x
    return z[0], z[1], np.append(z[2:], 100 - z[2:].sum())


k_CP_fit, k_GC_fit, x0 = estimasi()

# ---------------------------------------------------------------
# 3. Fungsi bantu tampilan
# ---------------------------------------------------------------
NAMA_BULAN = "Januari Februari Maret April Mei Juni Juli Agustus September Oktober November Desember".split()


def angka(v, n=1):
    return f"{v:.{n}f}".replace(".", ",")


def kalender(t_):
    m = int(t_) + 3                                    # t = 0 adalah April 2025
    return f"{NAMA_BULAN[m % 12]} {2025 + m // 12}"


def menyalip(t_, x_):
    i = np.where(x_[:, 2] >= x_[:, 0])[0]
    return t_[i[0]] if len(i) else None


def grafik(ax, judul, xlabel=None, kolom=3):
    ax.set(title=judul, xlabel=xlabel, ylabel="Pangsa pasar (%)", ylim=(0, 100))
    ax.title.set_fontweight("bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if xlabel is None:
        ax.set_xticks(range(len(BULAN)), BULAN, rotation=60, ha="right", fontsize=8)
    ax.legend(ncol=kolom, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    ax.figure.tight_layout()
    st.pyplot(ax.figure)


# ---------------------------------------------------------------
# 4. Parameter dan skenario (sidebar)
# ---------------------------------------------------------------
st.sidebar.header("Parameter Model")
st.sidebar.caption("Nilai awal adalah hasil estimasi dari data.")
OPSI = [round(0.0005 * i, 4) for i in range(601)]      # 0 sampai 0,3


def geser(label, v):
    return st.sidebar.select_slider(label, OPSI, round(round(v / 0.0005) * 0.0005, 4), format_func=lambda a: angka(a, 4))


k = (geser("k_CP: Perplexity → ChatGPT", k_CP_fit), geser("k_GC: ChatGPT → Gemini", k_GC_fit),
     geser("k_OC: ChatGPT → chatbot lainnya", 0))
t_max = st.sidebar.slider("Durasi simulasi (bulan)", 24, 72, 72, 6)

st.sidebar.header("Skenario Intervensi")
SKENARIO = {"Skenario dasar (tanpa intervensi)": (None, "Tanpa perubahan parameter."),
            "A: Penguatan integrasi ekosistem Gemini": (lambda a, b, c: (a, 2 * b, c), "k_GC × 2."),
            "B: Peningkatan retensi pengguna ChatGPT": (lambda a, b, c: (a, 0.5 * b, c), "k_GC × 0,5."),
            "C: Pemulihan pangsa Perplexity": (lambda a, b, c: (-a, b, c), "Tanda k_CP dibalik."),
            "D: Pertumbuhan chatbot penantang": (lambda a, b, c: (a, b, b), "k_OC = k_GC.")}
pilih = st.sidebar.radio("Pilih skenario", list(SKENARIO))
ubah, ket = SKENARIO[pilih]
st.sidebar.caption(ket + (" Berlaku mulai Oktober 2026." if ubah else ""))

t, x = simulasi(x0, k, t_max)
t_s, x_s = simulasi(x0, k, t_max, (18, ubah(*k)) if ubah else None)
nama_sk = f"skenario {pilih[0]}" if ubah else "skenario dasar"

tab1, tab2, tab3, tab4 = st.tabs(["Data", "Validasi Model", "Proyeksi dan Skenario", "Verifikasi"])

with tab1:
    st.markdown("Sumber: **StatCounter Global Stats**, April 2025 - November 2026. "
                "Data Oktober-November 2026 merupakan anomali dan tidak digunakan dalam estimasi.")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axvspan(17.5, 19.5, color="orange", alpha=0.15, label="Anomali data (Okt-Nov 2026)")
    for (nama, v), w in zip(DATA.items(), WARNA):
        ax.plot(v, "-o", ms=3.5, color=w, label=nama)
    grafik(ax, "Pangsa Pasar Chatbot AI di Indonesia", kolom=4)
    st.markdown("**Pangsa pasar (%)**")
    st.dataframe({"Bulan": BULAN, **{n: [angka(a, 2) for a in v] for n, v in DATA.items()}},
                 hide_index=True, height=35 * (len(BULAN) + 1) + 3)

with tab2:
    mae = np.abs(x[np.arange(N_FIT) * round(1 / dt)] - x_data[:N_FIT]).mean(0)
    st.markdown("Estimasi kuadrat terkecil dengan data April 2025 - September 2026. "
                "MAE dinyatakan dalam poin persentase. Semakin kecil nilainya, semakin sesuai model dengan data.")
    c1, c2 = st.columns(2)
    c1.metric("Estimasi k_CP (per bulan)", angka(k_CP_fit, 4))
    c2.metric("Estimasi k_GC (per bulan)", angka(k_GC_fit, 4))
    for kol, n, v in zip(st.columns(4), NAMA, mae):
        kol.metric(f"MAE {n}", f"{angka(v, 2)} poin")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axvspan(17.5, 19.5, color="orange", alpha=0.15, label="Anomali (tidak digunakan dalam estimasi)")
    for i in range(4):
        ax.plot(t[t <= 19], x[t <= 19, i], color=WARNA[i], lw=2, label=f"{NAMA[i]} (model)")
        ax.plot(x_data[:, i], "o", color=WARNA[i], ms=5, label=f"{NAMA[i]} (data)")
    grafik(ax, "Model dan Data StatCounter")

with tab3:
    ip, ts1, ts2 = int(np.argmax(x[:, 0])), menyalip(t, x), menyalip(t_s, x_s)
    d1, d2, d3 = st.columns(3)
    d1.metric("Pangsa puncak ChatGPT", f"{angka(x[ip, 0])}%")
    d1.caption(kalender(t[ip]))
    d2.metric("Gemini melampaui ChatGPT (skenario dasar)", kalender(ts1) if ts1 is not None else "Tidak terjadi")
    d2.caption(f"Bulan ke-{angka(ts1)}" if ts1 is not None else f"Dalam {t_max} bulan")
    if ubah:
        d3.metric(f"Gemini melampaui ChatGPT ({nama_sk})", kalender(ts2) if ts2 is not None else "Tidak terjadi")
        d3.caption(f"Bulan ke-{angka(ts2)}" if ts2 is not None else f"Dalam {t_max} bulan")
    else:
        d3.metric("Gemini melampaui ChatGPT (intervensi)", "-")
        d3.caption("Pilih skenario A-D")
    st.markdown(f"**Pangsa pasar {kalender(t_max)} ({nama_sk})**")
    for kol, n, v in zip(st.columns(4), NAMA, x_s[-1]):
        kol.metric(n, f"{angka(v)}%")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for i in range(4):
        ax.plot(t, x[:, i], color=WARNA[i], lw=2, label=f"{NAMA[i]} (skenario dasar)")
        if ubah:
            ax.plot(t_s, x_s[:, i], color=WARNA[i], lw=2, ls="--", label=f"{NAMA[i]} ({nama_sk})")
    ax.axvline(17, color="red", ls=":", label="Akhir periode estimasi (Sep 2026)")
    if ts1 is not None:
        ax.axvline(ts1, color="gray", ls="-.", alpha=0.7, label=f"Gemini melampaui ChatGPT ({kalender(ts1)})")
    ax.set_xlim(0, t_max)
    grafik(ax, "Proyeksi Pangsa Pasar Chatbot AI", "Bulan sejak April 2025")
    st.info("Proyeksi mengasumsikan parameter tetap dan bukan peramalan pasar.")

with tab4:
    rk = solve_ivp(lambda _, xx: turunan(xx, *k), (0, t_max), x0, t_eval=t[::100], rtol=1e-10, atol=1e-12).y.T
    x_half = simulasi(x0, k, t_max, h=dt / 2)[1]
    simpangan = np.abs(x.sum(1) - 100).max()
    st.markdown("Tiga pemeriksaan untuk memastikan perhitungan program benar. Selisih dinyatakan dalam poin persentase.")
    e1, e2, e3 = st.columns(3)
    e1.metric("Total pangsa tetap 100%", "Terpenuhi" if simpangan < 1e-6 else "Tidak terpenuhi")
    e1.caption("Simpangan < 10⁻¹² (galat pembulatan)" if simpangan < 1e-12 else f"Simpangan {angka(simpangan, 6)} poin")
    e2.metric("Selisih Euler dan Runge-Kutta", f"{angka(np.abs(x[::100] - rk).max(), 3)} poin")
    e2.caption("Selisih maksimum")
    e3.metric("Selisih saat Δt diperkecil", f"{angka(np.abs(x_half[::200] - x[::100]).max(), 3)} poin")
    e3.caption(f"Selisih maksimum saat Δt {angka(dt, 2)} → {angka(dt / 2, 3)} bulan")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i in range(4):
        ax.plot(t[::100], rk[:, i], color="black", lw=1.5)
        ax.plot(t, x[:, i], color=WARNA[i], ls="--", lw=2, label=f"{NAMA[i]} (Euler)")
    ax.plot([], [], color="black", label="Runge-Kutta")
    ax.set_xlim(0, t_max)
    grafik(ax, "Solusi Euler dan Runge-Kutta", "Bulan sejak April 2025")

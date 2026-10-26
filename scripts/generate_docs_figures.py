from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "AB_NYC_2019.csv"
IMAGE_DIR = ROOT / "docs" / "assets" / "images"


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(IMAGE_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {DATA_PATH}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    data = pd.read_csv(DATA_PATH)
    positive = data.loc[data["price"] > 0].copy()

    if positive.empty:
        raise ValueError("O dataset não contém preços positivos para gerar os gráficos.")

    p95 = positive["price"].quantile(0.95)
    p99 = positive["price"].quantile(0.99)
    median = positive["price"].median()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(positive["price"], bins=70, log_scale=(True, False), ax=ax, color="#00897b")
    ax.axvline(median, color="#ef6c00", linestyle="--", label=f"Mediana: US$ {median:.0f}")
    ax.axvline(p99, color="#455a64", linestyle=":", label=f"P99: US$ {p99:.0f}")
    ax.set(title="Distribuição dos preços anunciados", xlabel="Preço por noite (US$, escala log)", ylabel="Número de anúncios")
    ax.legend()
    save_figure(fig, "price_distribution.png")

    room_order = positive.groupby("room_type")["price"].median().sort_values(ascending=False).index
    room_counts = positive["room_type"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=positive,
        x="room_type",
        y="price",
        order=room_order,
        showfliers=False,
        color="#80cbc4",
        ax=ax,
    )
    ax.set_ylim(0, p99)
    ax.set(
        title="Preço por tipo de quarto (eixo limitado ao P99)",
        xlabel="Tipo de quarto",
        ylabel="Preço por noite (US$)",
    )
    ax.set_xticks(
        range(len(room_order)),
        labels=[f"{room}\n(n={room_counts[room]:,})" for room in room_order],
    )
    save_figure(fig, "price_room_type.png")

    borough_stats = (
        positive.groupby("neighbourhood_group")["price"]
        .agg(median="median", count="size")
        .sort_values("median", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        borough_stats.index,
        borough_stats["median"],
        color=sns.color_palette("crest", len(borough_stats)),
    )
    for bar, count in zip(bars, borough_stats["count"]):
        ax.annotate(
            f"n={count:,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set(
        title="Mediana do preço por distrito",
        xlabel="Distrito",
        ylabel="Mediana do preço por noite (US$)",
    )
    ax.tick_params(axis="x", rotation=15)
    save_figure(fig, "price_borough.png")

    map_sample = positive.sample(min(8000, len(positive)), random_state=42)
    fig, ax = plt.subplots(figsize=(8, 7))
    points = ax.scatter(
        map_sample["longitude"],
        map_sample["latitude"],
        c=map_sample["price"].clip(upper=p95),
        cmap="viridis",
        s=9,
        alpha=0.62,
        linewidths=0,
    )
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label(f"Preço por noite (US$; saturado no P95 = {p95:.0f})")
    ax.set(
        title="Localização dos anúncios e preço anunciado",
        xlabel="Longitude",
        ylabel="Latitude",
        aspect="equal",
    )
    save_figure(fig, "price_location.png")

    numeric_columns = [
        "latitude",
        "longitude",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]
    correlation = data[numeric_columns].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlação de Spearman entre variáveis numéricas")
    save_figure(fig, "numeric_correlation.png")

    missing = data.isna().mean().mul(100).sort_values(ascending=False)
    missing = missing.loc[missing > 0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(missing.index, missing.values, color="#00897b")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(
        title="Valores ausentes por coluna",
        xlabel="Coluna",
        ylabel="Registros ausentes (%)",
        ylim=(0, max(missing.values) * 1.18),
    )
    ax.tick_params(axis="x", rotation=25)
    save_figure(fig, "missingness.png")


if __name__ == "__main__":
    main()

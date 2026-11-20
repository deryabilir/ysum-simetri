<div class="section-card">
        Bu bölüm A–E ölçümlerini tek görsel üzerinde gösterir.
        Eğitilmiş sınıflandırma modeli daha sonra eklenecektir.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Model henüz eğitilmediği için nevüs/melanom tahmini veya "
        "risk skoru gösterilmez; yalnızca görüntü ölçümleri hesaplanır."
    )

    uploaded_model_image = st.file_uploader(
        "İncelenecek dermoskopik görsel",
        type=["jpg", "jpeg", "png", "webp"],
        key="ysum_model_input",
    )

    if uploaded_model_image is not None:
        try:
            model_image = image_from_bytes(
                uploaded_model_image.getvalue()
            )

            # Sabit A ayarları:
            # k=2, 180 açı, 4 px adım, merkez ayarı kapalı.
            model_analysis = gorsel_analiz(
                model_image,
                k=2,
                n_angles=180,
                step_px=4,
                center_adjust=False,
                with_heat=True,
            )
            model_rgb = model_analysis["rgb"]
            model_segmentation = segment_lesion(model_rgb)
            model_features = lesion_features(
                model_rgb,
                model_segmentation,
            )

            st.markdown("#### Görüntü ölçümleri")

            metric_columns = st.columns(5)
            metric_columns[0].metric(
                "A — Dönel U",
                fmt_number(model_analysis["U"]),
            )
            metric_columns[1].metric(
                "B — Kenar düzensizliği",
                fmt_number(model_features["B_edge"]),
            )
            metric_columns[2].metric(
                "C — Renk sınıfı sayısı",
                fmt_number(model_features["C_colors"], 0),
            )
            metric_columns[3].metric(
                "D — Gri ton entropisi (bit)",
                fmt_number(model_features["D_entropy_bits"], 3),
            )
            metric_columns[4].metric(
                "E — İç/çevre renk farkı",
                fmt_number(model_features["E_spatial_color_tv"], 3),
            )

            image_column, mask_column = st.columns(2)

            with image_column:
                st.image(
                    model_rgb,
                    caption="Analizde kullanılan 200×200 RGB görüntü",
                    use_container_width=True,
                )

            with mask_column:
                st.image(
                    mask_overlay_array(
                        model_rgb,
                        model_segmentation["mask"],
                    ),
                    caption=(
                        "Otsu’nun yaklaşık maskesi: renkli alan; "
                        "sarı çizgi: yaklaşık sınır"
                    ),
                    use_container_width=True,
                )

            show_ae_maps(
                model_analysis,
                model_segmentation,
                model_features,
            )
            st.caption(
                "A, k=2 (180°), 180 açı ve 4 px adımla hesaplanır. "
                "D, Otsu maskesindeki 32 gri ton seviyesinin entropisidir. "
                "E, maskenin iç ve sınıra yakın bölgelerindeki renk "
                "dağılımlarının farkıdır."
            )

            st.caption(
                "Otsu maskesinin geçerli bulunması, gerçek lezyon "
                "sınırını doğru çizdiğini kanıtlamaz. Mevcut ısı haritası "
                "kanser risk haritası değildir."
            )

            if not model_segmentation["valid"]:
                st.warning(
                    "Bu görüntüde Otsu maskesi geçerli bulunmadı. "
                    "B, C, D ve E maskeye bağlı ölçümler olduğundan "
                    "hesaplanamamış veya güvenilir olmayabilir."
                )

        except Exception as exc:
            st.error(f"Görsel analiz edilemedi: {exc}")

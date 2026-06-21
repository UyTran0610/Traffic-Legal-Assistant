import asyncio
import flet as ft
from config import SIDEBAR_COLOR, SIDEBAR_HOVER, USER_BUBBLE_COLOR
from backend import ai_state, session_manager, initialize_ai_system, retrieve_context

# ==========================================
# COMPONENT: MESSAGE BUBBLE
# ==========================================
class MessageBubble(ft.Container):
    def __init__(self, text, is_user=False, page_ref=None):
        super().__init__()
        self.is_user = is_user
        self._page_ref = page_ref
        self.text_control = ft.Markdown(
            text,
            selectable=True,
            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
            code_theme="atom-one-light",
            on_tap_link=lambda e: page_ref.launch_url(e.data),
        )

        # Avatar
        avatar = ft.Container(
            content=ft.Icon(
                ft.Icons.PERSON if is_user else ft.Icons.AUTO_AWESOME,
                color=ft.Colors.WHITE if is_user else ft.Colors.BLUE_600,
                size=20
            ),
            width=32, height=32,
            bgcolor=ft.Colors.BLUE_600 if is_user else ft.Colors.BLUE_50,
            border_radius=16,
            alignment=ft.Alignment(0, 0),
            margin=ft.margin.only(top=5)
        )

        # Nội dung tin nhắn
        BOT_BUBBLE_COLOR = ft.Colors.TRANSPARENT
        content_container = ft.Container(
            content=self.text_control,
            bgcolor=USER_BUBBLE_COLOR if is_user else BOT_BUBBLE_COLOR,
            border_radius=12,
            padding=ft.padding.symmetric(vertical=10, horizontal=15) if is_user else ft.padding.only(top=5),
            # Tin nhắn User có max width giới hạn, Bot full width
            width=None if not is_user else 600,
        )

        if is_user:
            self.content = ft.Row(
                [content_container, ft.Container(width=5), avatar],
                alignment=ft.MainAxisAlignment.END,
                vertical_alignment=ft.CrossAxisAlignment.START,
            )
        else:
            self.content = ft.Row(
                [avatar, ft.Container(width=10), ft.Container(content=content_container, expand=True)],
                alignment=ft.MainAxisAlignment.START,
                vertical_alignment=ft.CrossAxisAlignment.START,
                expand=True
            )

        self.padding = ft.padding.symmetric(vertical=10, horizontal=20)

    def update_text(self, new_text):
        self.text_control.value = new_text
        if self._page_ref:
            self._page_ref.update()
        else:
            self.update()


# ==========================================
# MAIN APP
# ==========================================
async def main(page: ft.Page):
    # Cấu hình chung cho Page
    page.title = "Trợ Lý Luật Giao Thông - AI"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0
    page.bgcolor = ft.Colors.WHITE
    page.fonts = {"Roboto": "Roboto"}

    # ==========================
    # UI COMPONENTS
    # ==========================

    # Biến tạm để lưu ID của session đang được thao tác
    target_session_id = {"id": None}

    _active_dialog = {"ref": None}

    def _close_active_dialog():
        """Đóng và xóa đúng dialog đang mở khỏi overlay."""
        dlg = _active_dialog["ref"]
        if dlg is not None and dlg in page.overlay:
            page.overlay.remove(dlg)
        _active_dialog["ref"] = None
        page.update()

    # ----------------------------------------
    # Dialog: Đổi tên phiên chat
    # ----------------------------------------
    def create_rename_dialog():
        txt_rename = ft.TextField(label="Tên mới", autofocus=True, value="", border_color=ft.Colors.BLUE)

        def confirm_rename_action(e):
            if target_session_id["id"] and txt_rename.value:
                session_manager.rename_session(target_session_id["id"], txt_rename.value)
                _close_active_dialog()
                load_sidebar_items()

        def cancel_rename_action(e):
            _close_active_dialog()

        # Tạo custom dialog bằng Container
        dialog_content = ft.Container(
            content=ft.Column([
                ft.Text("Đổi tên đoạn chat", size=20, weight=ft.FontWeight.BOLD),
                ft.Container(height=10),
                txt_rename,
                ft.Container(height=20),
                ft.Row([
                    ft.TextButton("Hủy", on_click=cancel_rename_action),
                    ft.ElevatedButton("Lưu", on_click=confirm_rename_action),
                ], alignment=ft.MainAxisAlignment.END),
            ], tight=True),
            width=400,
            padding=20,
            bgcolor=ft.Colors.WHITE,
            border_radius=10,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=15,
                color=ft.Colors.BLACK26,
            ),
        )

        # Backdrop (nền đen mờ)
        def close_on_backdrop_click(e):
            _close_active_dialog()

        backdrop = ft.Container(
            content=ft.Stack([
                # Background overlay
                ft.Container(
                    bgcolor=ft.Colors.BLACK54,
                    expand=True,
                    on_click=close_on_backdrop_click,
                ),
                # Dialog centered
                ft.Container(
                    content=dialog_content,
                    alignment=ft.alignment.Alignment(0, 0),
                    expand=True,
                ),
            ]),
            expand=True,
        )

        return backdrop, txt_rename

    # ----------------------------------------
    # Dialog: Xác nhận xóa phiên chat
    # ----------------------------------------
    def create_delete_dialog():
        def confirm_delete_action(e):
            sid = target_session_id["id"]
            if sid:
                session_manager.delete_session(sid)

                sessions = session_manager.get_all_sessions()
                if not sessions:
                    session_manager.create_session()
                elif session_manager.current_session_id is None:
                    session_manager.switch_session(sessions[0][0])

                _close_active_dialog()
                load_sidebar_items()
                load_chat_window()

        def cancel_delete_action(e):
            _close_active_dialog()

        # Tạo custom dialog
        dialog_content = ft.Container(
            content=ft.Column([
                ft.Text("Xác nhận xóa", size=20, weight=ft.FontWeight.BOLD),
                ft.Container(height=10),
                ft.Text("Bạn có chắc chắn muốn xóa đoạn chat này không?", size=14),
                ft.Container(height=20),
                ft.Row([
                    ft.TextButton("Hủy", on_click=cancel_delete_action),
                    ft.ElevatedButton(
                        "Xóa",
                        on_click=confirm_delete_action,
                        bgcolor=ft.Colors.RED,
                        color=ft.Colors.WHITE,
                    ),
                ], alignment=ft.MainAxisAlignment.END),
            ], tight=True),
            width=400,
            padding=20,
            bgcolor=ft.Colors.WHITE,
            border_radius=10,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=15,
                color=ft.Colors.BLACK26,
            ),
        )

        # Backdrop
        def close_on_backdrop_click(e):
            _close_active_dialog()

        backdrop = ft.Container(
            content=ft.Stack([
                ft.Container(
                    bgcolor=ft.Colors.BLACK54,
                    expand=True,
                    on_click=close_on_backdrop_click,
                ),
                ft.Container(
                    content=dialog_content,
                    alignment=ft.alignment.Alignment(0, 0),
                    expand=True,
                ),
            ]),
            expand=True,
        )

        return backdrop

    # Hàm mở dialog (sẽ được gọi từ nút 3 chấm)
    def open_rename(sid, current_title):
        target_session_id["id"] = sid
        _close_active_dialog()
        dialog, txt_field = create_rename_dialog()
        txt_field.value = current_title
        _active_dialog["ref"] = dialog
        page.overlay.append(dialog)
        page.update()

    def open_delete(sid):
        target_session_id["id"] = sid
        _close_active_dialog()
        dialog = create_delete_dialog()
        _active_dialog["ref"] = dialog
        page.overlay.append(dialog)
        page.update()

    # ----------------------------------------
    # SIDEBAR
    # ----------------------------------------
    sidebar_list = ft.ListView(expand=True, spacing=2, padding=10)

    # Hàm xử lý click menu items
    def handle_menu_click(e):
        if e.control.data:
            action = e.control.data.get("action")
            sid = e.control.data.get("sid")

            print(f"Menu item clicked: action={action}, sid={sid}")

            if action == "rename":
                title = e.control.data.get("title")
                open_rename(sid, title)
            elif action == "delete":
                open_delete(sid)

    def load_sidebar_items():
        sidebar_list.controls.clear()

        # Lấy danh sách sessions
        sessions = session_manager.get_all_sessions()

        for sid, title in sessions:
            is_active = (sid == session_manager.current_session_id)

            # --- TẠO MENU BUTTON ---
            menu_items = [
                ft.PopupMenuItem(
                    content=ft.Text("Đổi tên"),
                    data={"action": "rename", "sid": sid, "title": title},
                    on_click=handle_menu_click
                ),
                ft.PopupMenuItem(
                    content=ft.Text("Xóa"),
                    data={"action": "delete", "sid": sid},
                    on_click=handle_menu_click
                ),
            ]

            # Nút ⋮ dọc - mặc định ẩn
            menu_button = ft.PopupMenuButton(
                icon=ft.Icons.MORE_VERT,
                icon_color=ft.Colors.GREY_600,
                icon_size=14,
                items=menu_items,
                visible=False,
                style=ft.ButtonStyle(
                    padding=ft.padding.all(0),
                    shape=ft.RoundedRectangleBorder(radius=4),
                ),
            )

            bg_container = ft.Container(
                padding=ft.padding.only(left=10, right=2, top=6, bottom=6),
                border_radius=8,
                bgcolor=SIDEBAR_HOVER if is_active else ft.Colors.TRANSPARENT,
                height=38,
                clip_behavior=ft.ClipBehavior.HARD_EDGE,
            )

            def show_menu(mb=menu_button, bg=bg_container):
                mb.visible = True
                bg.bgcolor = SIDEBAR_HOVER
                bg.update()

            def hide_menu(mb=menu_button, bg=bg_container, active=is_active):
                mb.visible = False
                bg.bgcolor = SIDEBAR_HOVER if active else ft.Colors.TRANSPARENT
                bg.update()

            row_content = ft.Row(
                controls=[
                    ft.Icon(ft.Icons.CHAT_BUBBLE_OUTLINE, size=16, color=ft.Colors.GREY_600),
                    ft.Container(
                        content=ft.Text(
                            title, size=14, color=ft.Colors.BLACK87,
                            no_wrap=True, overflow=ft.TextOverflow.ELLIPSIS,
                        ),
                        expand=True,
                        on_click=lambda e, s=sid: switch_chat(s),
                    ),
                    menu_button,
                ],
                spacing=0,
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )

            bg_container.content = ft.GestureDetector(
                content=row_content,
                on_enter=lambda e, sm=show_menu: sm(),
                on_exit=lambda e, hm=hide_menu: hm(),
                mouse_cursor=ft.MouseCursor.CLICK,
            )

            sidebar_list.controls.append(bg_container)

        if sidebar_list.page:
            sidebar_list.update()

    def switch_chat(sid):
        session_manager.switch_session(sid)
        load_sidebar_items()
        load_chat_window()

    def new_chat_click(e):
        session_manager.create_session()
        load_sidebar_items()
        load_chat_window()

    btn_new_chat = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.ADD, color=ft.Colors.BLUE_700),
            ft.Text("Cuộc trò chuyện mới", color=ft.Colors.BLUE_700, weight=ft.FontWeight.W_500)
        ]),
        padding=12,
        border_radius=8,
        border=ft.border.all(1, ft.Colors.BLUE_100),
        on_click=new_chat_click,
        ink=True,
        margin=ft.margin.only(bottom=10)
    )

    sidebar = ft.Container(
        width=260,
        bgcolor=SIDEBAR_COLOR,
        padding=10,
        content=ft.Column([
            btn_new_chat,
            ft.Divider(height=1, color=ft.Colors.GREY_300),
            ft.Text("Gần đây", size=12, color=ft.Colors.GREY_600, weight=ft.FontWeight.BOLD),
            sidebar_list
        ]),
        border=ft.border.only(right=ft.border.BorderSide(1, ft.Colors.GREY_300))
    )

    # ----------------------------------------
    # CHAT WINDOW
    # ----------------------------------------
    chat_list_view = ft.ListView(
        expand=True,
        spacing=0,
        padding=ft.padding.only(bottom=20),
        auto_scroll=True,
    )

    def load_chat_window():
        chat_list_view.controls.clear()
        messages = session_manager.get_messages()

        # Tin nhắn chào mừng nếu chưa có tin nhắn
        if not messages:
            chat_list_view.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.Icons.GAVEL, size=60, color=ft.Colors.BLUE_200),
                        ft.Text("Trợ Lý Luật Giao Thông", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.BLACK87),
                        ft.Text("Tôi có thể giúp gì cho bạn hôm nay?", color=ft.Colors.GREY_600)
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    alignment=ft.Alignment(0, 0),
                    padding=ft.padding.only(top=100)
                )
            )
        else:
            for msg in messages:
                bubble = MessageBubble(msg['text'], is_user=(msg['role'] == "user"), page_ref=page)
                chat_list_view.controls.append(bubble)

        chat_list_view.update()

    # ----------------------------------------
    # INPUT AREA
    # ----------------------------------------
    txt_input = ft.TextField(
        hint_text="Hỏi về luật giao thông (VD: Vượt đèn đỏ phạt bao nhiêu?)...",
        border=ft.InputBorder.NONE,
        filled=False,
        expand=True,
        multiline=True,
        max_lines=5,
        min_lines=1,
        shift_enter=True,
        on_submit=lambda e: send_message_action(None),
        text_style=ft.TextStyle(size=16),
        content_padding=15
    )

    _is_processing = {"value": False}

    def send_message_action(e):
        if _is_processing["value"]:
            return

        question = txt_input.value.strip()
        if not question: return

        _is_processing["value"] = True
        txt_input.value = ""
        txt_input.disabled = True
        btn_send.disabled = True
        page.update()

        # Snapshot session_id ngay tại thời điểm gửi — tránh race condition:
        # nếu người dùng chuyển session trong lúc AI đang stream, mọi thao tác
        # ghi DB của task này vẫn nhắm đúng session gốc.
        active_session_id = session_manager.current_session_id
        if not active_session_id:
            active_session_id = session_manager.create_session()

        # Nếu là session mới tinh, clear giao diện welcome
        if not session_manager.get_messages(active_session_id):
            chat_list_view.controls.clear()

        # Thêm user message vào UI và Data
        session_manager.add_message("user", question, session_id=active_session_id)
        user_bubble = MessageBubble(question, is_user=True, page_ref=page)
        chat_list_view.controls.append(user_bubble)

        # Cập nhật sidebar để hiển thị title mới
        load_sidebar_items()

        # Tạo Bot Bubble placeholder
        bot_bubble = MessageBubble("⚪ Đang tra cứu luật...", is_user=False, page_ref=page)
        chat_list_view.controls.append(bot_bubble)
        page.update()

        # Lấy toàn bộ messages sau khi đã add câu hỏi user
        current_messages = session_manager.get_messages(active_session_id)
        page.run_task(process_ai_response, question, bot_bubble, active_session_id, current_messages)

    def _unlock_input():
        """Re-enable input sau khi AI trả lời xong (cả thành công lẫn lỗi)."""
        _is_processing["value"] = False
        txt_input.disabled = False
        btn_send.disabled = False
        page.update()

    async def process_ai_response(question, bot_bubble, session_id, current_messages=None):
        loop = asyncio.get_event_loop()
        try:
            # Chạy các hàm blocking trong executor để không block event loop
            context, legal_hint, chat_history_str = await loop.run_in_executor(
                None, lambda: retrieve_context(question, top_k=5, messages=current_messages or [])
            )

            final_text = ""
            if not context:
                final_text = "🤖 Xin lỗi, tôi không tìm thấy quy định phù hợp trong cơ sở dữ liệu."
                bot_bubble.text_control.value = final_text
                page.update()
                print("   [Log] Không tìm thấy context, trả về câu mặc định.")
            else:
                print("   🤖 Bot đang sinh câu trả lời (Streaming)...")
                inputs = {
                    "context": context,
                    "question": question,
                    "legal_hint": legal_hint,
                    "chat_history": chat_history_str,
                }

                # --- SỬ DỤNG NATIVE ASYNC STREAMING CỦA LANGCHAIN (ASTREAM) ---
                # Nếu astream ném exception, outer except bên dưới sẽ bắt và xử lý.
                async for chunk in ai_state.llm_chain.astream(inputs):
                    final_text += chunk
                    bot_bubble.text_control.value = final_text + " ▌"
                    page.update()

                bot_bubble.text_control.value = final_text
                page.update()
                print("   ✅ Đã trả lời xong.")

            # Lưu tin nhắn Bot vào đúng session gốc (session_id được snapshot
            # từ lúc gửi, không bị ảnh hưởng dù người dùng đã chuyển session)
            session_manager.add_message("bot", final_text, session_id=session_id)

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"❌ Có lỗi xảy ra: {str(e)}"
            bot_bubble.text_control.value = error_msg
            page.update()
            # Lưu cả tin nhắn lỗi vào DB — tránh conversation bị treo
            # (user message có rồi nhưng bot reply không có → hỏng history khi reload)
            session_manager.add_message("bot", error_msg, session_id=session_id)
            print(error_msg)
        finally:
            # Luôn mở khóa input, dù thành công hay lỗi
            _unlock_input()

    btn_send = ft.IconButton(
        icon=ft.Icons.ARROW_UPWARD_ROUNDED,
        icon_color=ft.Colors.WHITE,
        bgcolor=ft.Colors.BLACK87,
        width=36, height=36,
        on_click=lambda e: send_message_action(None),
        tooltip="Gửi tin nhắn"
    )

    input_container = ft.Container(
        content=ft.Row(
            [txt_input, btn_send],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.END
        ),
        padding=8,
        border_radius=26,
        bgcolor=ft.Colors.GREY_100,
        margin=ft.margin.symmetric(horizontal=50, vertical=15),
    )

    # ----------------------------------------
    # LOADING OVERLAY
    # ----------------------------------------
    loading_text = ft.Text("Đang khởi động hệ thống AI (Loading models)...", color=ft.Colors.BLUE_700, size=16)
    loading_overlay = ft.Container(
        content=ft.Column([
            ft.ProgressRing(width=50, height=50, stroke_width=4, color=ft.Colors.BLUE_700),
            ft.Container(height=20),
            loading_text
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        alignment=ft.Alignment(0, 0),
        bgcolor=ft.Colors.WHITE,
        expand=True,
    )

    # ----------------------------------------
    # LAYOUT CHÍNH (Split View)
    # ----------------------------------------
    right_column = ft.Column([
        chat_list_view,
        ft.Container(
            content=input_container,
            bgcolor=ft.Colors.WHITE,  # Nền trắng cho thanh input
        )
    ], expand=True, spacing=0)

    main_layout = ft.Row(
        [
            sidebar,
            ft.VerticalDivider(width=1, color=ft.Colors.GREY_300),
            right_column
        ],
        expand=True,
        spacing=0
    )

    page.add(loading_overlay)

    async def check_ai_ready():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, initialize_ai_system)

        if ai_state.init_error:
            loading_text.value = f"Lỗi: {ai_state.init_error}. Vui lòng kiểm tra lại!"
            loading_text.color = ft.Colors.RED
            page.update()
            return

        existing_sessions = session_manager.get_all_sessions()
        if existing_sessions:
            session_manager.switch_session(existing_sessions[0][0])
        else:
            session_manager.create_session()

        page.clean()
        page.add(main_layout)
        load_sidebar_items()
        load_chat_window()
        page.update()

    page.run_task(check_ai_ready)


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    ft.app(target=main)
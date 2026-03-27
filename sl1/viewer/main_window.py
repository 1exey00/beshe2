"""策略查看器主窗口模块。

本模块实现了策略查看器的主窗口，组合所有UI组件：
- 游戏树控件
- 手牌矩阵控件
- 信息面板控件
- 策略详情控件
- 公共牌选择器
- 菜单栏（文件加载、导出等）

需求引用:
- 需求 7.3: 用户调整窗口大小时自适应调整布局
- 需求 7.4: 用户进行任何操作时提供视觉反馈
"""

from typing import Optional, List
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QTabWidget,
    QMenuBar,
    QMenu,
    QStatusBar,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
    QApplication,
    QLabel,
    QFrame,
    QPushButton,
    QScrollArea,
    QGroupBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QFont
from models.core import ActionType

from models.core import Card, GameStage
from viewer.controller import StrategyViewerController
from viewer.models import GameTreeNode, NodeState, HandStrategy, NodeType
from viewer.model_loader import ModelMetadata
from viewer.strategy_calculator import StrategyResult
from viewer.widgets import (
    GameTreeWidget,
    HandRangeMatrixWidget,
    InfoPanelWidget,
    StrategyDetailWidget,
    BoardCardSelector,
    ValueHeatmapWidget,
    RegretHeatmapWidget,
)


class MainWindow(QMainWindow):
    """策略查看器主窗口 - 组合所有UI组件。
    
    该窗口包含以下主要区域：
    1. 左侧面板：游戏树导航
    2. 中央面板：手牌范围矩阵
    3. 右侧面板：信息面板和策略详情（标签页切换）
    4. 菜单栏：文件操作、导出等
    5. 状态栏：显示操作反馈
    
    Attributes:
        _controller: 策略查看器控制器
        _game_tree_widget: 游戏树控件
        _hand_matrix_widget: 手牌矩阵控件
        _info_panel_widget: 信息面板控件
        _strategy_detail_widget: 策略详情控件
        _board_selector_widget: 公共牌选择器控件
    """
    
    # 窗口默认大小
    DEFAULT_WIDTH = 1400
    DEFAULT_HEIGHT = 900
    
    # 窗口最小大小
    MIN_WIDTH = 1000
    MIN_HEIGHT = 700
    
    def __init__(self, parent: Optional[QWidget] = None):
        """初始化主窗口。
        
        Args:
            parent: 父控件
        """
        super().__init__(parent)
        
        # 初始化控制器
        self._controller = StrategyViewerController()
        
        # 设置窗口属性
        self._setup_window()
        
        # 创建UI组件
        self._setup_ui()
        
        # 创建菜单栏
        self._setup_menu_bar()
        
        # 创建状态栏
        self._setup_status_bar()
        
        # 连接信号
        self._connect_signals()
        
        # 设置控制器回调
        self._setup_controller_callbacks()
        
        # 初始化显示
        self._initialize_display()

    def _setup_window(self) -> None:
        """设置窗口基本属性。"""
        self.setWindowTitle("策略查看器 - Texas Hold'em AI")
        self.setMinimumSize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)
        
        # 设置窗口图标（如果有的话）
        # self.setWindowIcon(QIcon("path/to/icon.png"))
    
    def _setup_ui(self) -> None:
        """设置UI布局。
        
        新布局结构：
        - 顶部：路径显示 + 行动按钮
        - 下方：手牌矩阵（左） + 信息面板（右）
        """
        # 创建中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局（垂直）
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # 顶部面板：路径和行动按钮
        top_panel = self._create_top_panel()
        main_layout.addWidget(top_panel)
        
        # 下方分割器（水平分割）
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(bottom_splitter, 1)
        
        # 左侧面板：手牌矩阵
        left_panel = self._create_matrix_panel()
        bottom_splitter.addWidget(left_panel)
        
        # 右侧面板：信息和详情
        right_panel = self._create_right_panel()
        bottom_splitter.addWidget(right_panel)
        
        # 设置分割器比例
        bottom_splitter.setSizes([750, 400])
        
        # 保存分割器引用
        self._main_splitter = bottom_splitter
        
        # 创建隐藏的游戏树控件（用于内部导航）
        self._game_tree_widget = GameTreeWidget(
            navigator=self._controller._game_tree
        )
        self._game_tree_widget.hide()

    def _create_top_panel(self) -> QWidget:
        """创建顶部面板（路径和行动按钮）。"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 5px;
            }
        """)
        panel.setFixedHeight(100)
        
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # 左侧：路径显示和返回按钮
        path_widget = QWidget()
        path_layout = QVBoxLayout(path_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.setSpacing(5)
        
        # 路径标签
        self._path_label = QLabel("游戏开始")
        self._path_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
            }
        """)
        path_layout.addWidget(self._path_label)
        
        # 返回按钮
        back_btn = QPushButton("← 返回上一步")
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                color: white;
                border: none;
                padding: 5px 15px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        back_btn.clicked.connect(self._on_back_clicked)
        path_layout.addWidget(back_btn)
        
        layout.addWidget(path_widget)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setStyleSheet("background-color: #555;")
        layout.addWidget(separator)
        
        # 右侧：行动按钮区域
        actions_widget = QWidget()
        actions_layout = QVBoxLayout(actions_widget)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(5)
        
        # 当前阶段和玩家信息
        self._stage_label = QLabel("翻牌前 - P1")
        self._stage_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #00ff00;
                font-weight: bold;
            }
        """)
        actions_layout.addWidget(self._stage_label)
        
        # 行动按钮容器
        self._action_buttons_widget = QWidget()
        self._action_buttons_layout = QHBoxLayout(self._action_buttons_widget)
        self._action_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._action_buttons_layout.setSpacing(10)
        actions_layout.addWidget(self._action_buttons_widget)
        
        layout.addWidget(actions_widget, 1)
        
        # 初始化行动按钮
        self._update_action_buttons()
        
        return panel
    
    def _create_matrix_panel(self) -> QWidget:
        """创建手牌矩阵面板。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标题
        title_label = QLabel("手牌范围矩阵")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: #f5f5f5;
                border-bottom: 1px solid #ddd;
            }
        """)
        layout.addWidget(title_label)
        
        # 手牌矩阵控件
        self._hand_matrix_widget = HandRangeMatrixWidget()
        layout.addWidget(self._hand_matrix_widget, 1)
        
        # 图例
        legend_widget = self._create_legend_widget()
        layout.addWidget(legend_widget)
        
        return panel

    def _create_legend_widget(self) -> QWidget:
        """创建图例控件。"""
        widget = QFrame()
        widget.setStyleSheet("""
            QFrame {
                background-color: #fafafa;
                border-top: 1px solid #ddd;
                padding: 5px;
            }
        """)
        
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # 图例项
        legend_items = [
            ("加注/跟注", "#4CAF50"),
            ("弃牌", "#F44336"),
            ("混合策略", "#FFC107"),
            ("禁用", "#E0E0E0"),
        ]
        
        for text, color in legend_items:
            item_layout = QHBoxLayout()
            item_layout.setSpacing(5)
            
            # 颜色方块
            color_box = QLabel()
            color_box.setFixedSize(16, 16)
            color_box.setStyleSheet(f"""
                QLabel {{
                    background-color: {color};
                    border: 1px solid #999;
                    border-radius: 2px;
                }}
            """)
            item_layout.addWidget(color_box)
            
            # 文字
            text_label = QLabel(text)
            text_label.setStyleSheet("font-size: 11px; color: #666;")
            item_layout.addWidget(text_label)
            
            layout.addLayout(item_layout)
        
        layout.addStretch()
        
        return widget
    
    def _create_right_panel(self) -> QWidget:
        """创建右侧面板（信息和详情）。"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建标签页控件
        tab_widget = QTabWidget()
        self._right_tab_widget = tab_widget
        
        # 信息面板标签页
        self._info_panel_widget = InfoPanelWidget()
        tab_widget.addTab(self._info_panel_widget, "游戏信息")
        
        # 策略详情标签页
        self._strategy_detail_widget = StrategyDetailWidget()
        tab_widget.addTab(self._strategy_detail_widget, "策略详情")
        
        # 公共牌选择器标签页
        self._board_selector_widget = BoardCardSelector()
        tab_widget.addTab(self._board_selector_widget, "公共牌设置")
        
        # 遗憾值热图标签页（Deep CFR架构）
        self._regret_heatmap_widget = RegretHeatmapWidget()
        tab_widget.addTab(self._regret_heatmap_widget, "遗憾值估计")
        
        # 价值热图标签页（旧架构，保留兼容性）
        self._value_heatmap_widget = ValueHeatmapWidget()
        tab_widget.addTab(self._value_heatmap_widget, "价值估计(旧)")
        
        layout.addWidget(tab_widget)
        
        return panel

    def _setup_menu_bar(self) -> None:
        """设置菜单栏。"""
        menu_bar = self.menuBar()
        
        # 文件菜单
        file_menu = menu_bar.addMenu("文件(&F)")
        self._setup_file_menu(file_menu)
        
        # 视图菜单
        view_menu = menu_bar.addMenu("视图(&V)")
        self._setup_view_menu(view_menu)
        
        # 导出菜单
        export_menu = menu_bar.addMenu("导出(&E)")
        self._setup_export_menu(export_menu)
        
        # 帮助菜单
        help_menu = menu_bar.addMenu("帮助(&H)")
        self._setup_help_menu(help_menu)
    
    def _setup_file_menu(self, menu: QMenu) -> None:
        """设置文件菜单。"""
        # 加载检查点
        load_action = QAction("加载检查点(&L)...", self)
        load_action.setShortcut(QKeySequence("Ctrl+O"))
        load_action.setStatusTip("加载训练好的模型检查点文件")
        load_action.triggered.connect(self._on_load_checkpoint)
        menu.addAction(load_action)
        
        # 最近文件（预留）
        # recent_menu = menu.addMenu("最近文件(&R)")
        
        menu.addSeparator()
        
        # 卸载模型
        unload_action = QAction("卸载模型(&U)", self)
        unload_action.setStatusTip("卸载当前加载的模型")
        unload_action.triggered.connect(self._on_unload_model)
        menu.addAction(unload_action)
        self._unload_action = unload_action
        
        menu.addSeparator()
        
        # 退出
        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.setStatusTip("退出策略查看器")
        exit_action.triggered.connect(self.close)
        menu.addAction(exit_action)
    
    def _setup_view_menu(self, menu: QMenu) -> None:
        """设置视图菜单。"""
        # 展开所有节点
        expand_action = QAction("展开所有节点(&E)", self)
        expand_action.setShortcut(QKeySequence("Ctrl+E"))
        expand_action.triggered.connect(self._on_expand_all)
        menu.addAction(expand_action)
        
        # 折叠所有节点
        collapse_action = QAction("折叠所有节点(&C)", self)
        collapse_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        collapse_action.triggered.connect(self._on_collapse_all)
        menu.addAction(collapse_action)
        
        menu.addSeparator()
        
        # 重置到根节点
        reset_action = QAction("重置到根节点(&R)", self)
        reset_action.setShortcut(QKeySequence("Ctrl+Home"))
        reset_action.triggered.connect(self._on_reset_to_root)
        menu.addAction(reset_action)
        
        menu.addSeparator()
        
        # 刷新显示
        refresh_action = QAction("刷新显示(&F)", self)
        refresh_action.setShortcut(QKeySequence("F5"))
        refresh_action.triggered.connect(self._on_refresh)
        menu.addAction(refresh_action)

    def _setup_export_menu(self, menu: QMenu) -> None:
        """设置导出菜单。"""
        # 导出图片
        export_image_action = QAction("导出矩阵图片(&I)...", self)
        export_image_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        export_image_action.setStatusTip("将手牌矩阵导出为图片文件")
        export_image_action.triggered.connect(self._on_export_image)
        menu.addAction(export_image_action)
        self._export_image_action = export_image_action
        
        # 导出JSON
        export_json_action = QAction("导出策略JSON(&J)...", self)
        export_json_action.setStatusTip("将策略数据导出为JSON文件")
        export_json_action.triggered.connect(self._on_export_json)
        menu.addAction(export_json_action)
        self._export_json_action = export_json_action
    
    def _setup_help_menu(self, menu: QMenu) -> None:
        """设置帮助菜单。"""
        # 关于
        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(self._on_about)
        menu.addAction(about_action)
        
        # 使用说明
        help_action = QAction("使用说明(&H)", self)
        help_action.setShortcut(QKeySequence("F1"))
        help_action.triggered.connect(self._on_help)
        menu.addAction(help_action)
    
    def _setup_status_bar(self) -> None:
        """设置状态栏。"""
        status_bar = self.statusBar()
        
        # 模型状态标签
        self._model_status_label = QLabel("未加载模型")
        self._model_status_label.setStyleSheet("padding: 0 10px;")
        status_bar.addPermanentWidget(self._model_status_label)
        
        # 显示初始消息
        status_bar.showMessage("就绪 - 请加载检查点文件开始分析", 5000)
    
    def _connect_signals(self) -> None:
        """连接UI组件的信号。"""
        # 游戏树节点选择
        self._game_tree_widget.node_selected.connect(self._on_node_selected)
        
        # 手牌矩阵悬停和点击
        self._hand_matrix_widget.cell_hovered.connect(self._on_hand_hovered)
        self._hand_matrix_widget.cell_clicked.connect(self._on_hand_clicked)
        
        # 公共牌选择器
        self._board_selector_widget.board_changed.connect(self._on_board_changed)
        
        # 价值热图悬停和点击
        self._value_heatmap_widget.cell_hovered.connect(self._on_value_hand_hovered)
        self._value_heatmap_widget.cell_clicked.connect(self._on_value_hand_clicked)
        
        # 遗憾值热图悬停和点击
        self._regret_heatmap_widget.cell_hovered.connect(self._on_regret_hand_hovered)
        self._regret_heatmap_widget.cell_clicked.connect(self._on_regret_hand_clicked)

    def _setup_controller_callbacks(self) -> None:
        """设置控制器回调函数。"""
        self._controller.set_on_model_loaded(self._on_model_loaded_callback)
        self._controller.set_on_node_changed(self._on_node_changed_callback)
        self._controller.set_on_strategy_updated(self._on_strategy_updated_callback)
        self._controller.set_on_board_changed(self._on_board_changed_callback)
        self._controller.set_on_value_updated(self._on_value_updated_callback)
        self._controller.set_on_regret_updated(self._on_regret_updated_callback)
    
    def _initialize_display(self) -> None:
        """初始化显示状态。"""
        # 更新菜单状态
        self._update_menu_state()
        
        # 显示初始节点信息
        node = self._controller.get_current_node()
        if node:
            self._update_node_display(node)
            self._update_action_buttons()
    
    def _update_menu_state(self) -> None:
        """更新菜单项的启用状态。"""
        is_loaded = self._controller.is_model_loaded()
        
        self._unload_action.setEnabled(is_loaded)
        self._export_image_action.setEnabled(True)  # 即使没有模型也可以导出
        self._export_json_action.setEnabled(True)
    
    # ========================================================================
    # 菜单操作处理
    # ========================================================================
    
    def _on_load_checkpoint(self) -> None:
        """处理加载检查点操作。"""
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择检查点文件",
            str(Path.cwd() / "checkpoints"),
            "检查点文件 (*.pt *.pth);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        # 显示加载进度
        self.statusBar().showMessage(f"正在加载: {file_path}...")
        QApplication.processEvents()
        
        try:
            # 加载模型
            metadata = self._controller.load_model(file_path)
            
            # 显示成功消息
            self.statusBar().showMessage(
                f"模型加载成功 - 迭代次数: {metadata.episode_number}", 
                5000
            )
            
            # 更新模型状态标签
            self._model_status_label.setText(
                f"已加载: {Path(file_path).name}"
            )
            
        except Exception as e:
            # 显示错误消息
            QMessageBox.critical(
                self,
                "加载失败",
                f"无法加载检查点文件:\n{str(e)}"
            )
            self.statusBar().showMessage("加载失败", 3000)
        
        # 更新菜单状态
        self._update_menu_state()

    def _on_unload_model(self) -> None:
        """处理卸载模型操作。"""
        if not self._controller.is_model_loaded():
            return
        
        # 确认对话框
        reply = QMessageBox.question(
            self,
            "确认卸载",
            "确定要卸载当前模型吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._controller.unload_model()
            self._model_status_label.setText("未加载模型")
            self.statusBar().showMessage("模型已卸载", 3000)
            self._update_menu_state()
    
    def _on_expand_all(self) -> None:
        """处理展开所有节点操作。"""
        self._game_tree_widget.expand_all_nodes()
        self.statusBar().showMessage("已展开所有节点", 2000)
    
    def _on_collapse_all(self) -> None:
        """处理折叠所有节点操作。"""
        self._game_tree_widget.collapse_all_nodes()
        self.statusBar().showMessage("已折叠所有节点", 2000)
    
    def _on_reset_to_root(self) -> None:
        """处理重置到根节点操作。"""
        self._controller.reset_to_root()
        self._game_tree_widget.select_node(self._controller.get_root_node())
        self.statusBar().showMessage("已重置到根节点", 2000)
    
    def _on_refresh(self) -> None:
        """处理刷新显示操作。"""
        self._game_tree_widget.refresh()
        self._hand_matrix_widget.refresh()
        self._info_panel_widget.refresh()
        self._strategy_detail_widget.refresh()
        self.statusBar().showMessage("显示已刷新", 2000)
    
    def _on_export_image(self) -> None:
        """处理导出图片操作。"""
        # 打开保存文件对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出矩阵图片",
            str(Path.cwd() / "strategy_matrix.png"),
            "PNG图片 (*.png);;JPEG图片 (*.jpg);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # 导出图片
            saved_path = self._controller.export_image(file_path)
            
            QMessageBox.information(
                self,
                "导出成功",
                f"图片已保存到:\n{saved_path}"
            )
            self.statusBar().showMessage(f"图片已导出: {saved_path}", 5000)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "导出失败",
                f"无法导出图片:\n{str(e)}"
            )

    def _on_export_json(self) -> None:
        """处理导出JSON操作。"""
        # 打开保存文件对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出策略JSON",
            str(Path.cwd() / "strategy_data.json"),
            "JSON文件 (*.json);;所有文件 (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            # 导出JSON
            saved_path = self._controller.export_json(file_path)
            
            QMessageBox.information(
                self,
                "导出成功",
                f"策略数据已保存到:\n{saved_path}"
            )
            self.statusBar().showMessage(f"JSON已导出: {saved_path}", 5000)
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "导出失败",
                f"无法导出JSON:\n{str(e)}"
            )
    
    def _on_about(self) -> None:
        """处理关于对话框。"""
        QMessageBox.about(
            self,
            "关于策略查看器",
            """<h3>策略查看器</h3>
            <p>Texas Hold'em AI 策略分析工具</p>
            <p>版本: 1.0.0</p>
            <p>用于可视化和分析德州扑克AI模型的策略分布。</p>
            <hr>
            <p>功能特性:</p>
            <ul>
                <li>加载训练好的模型检查点</li>
                <li>游戏树导航浏览决策节点</li>
                <li>13x13手牌范围矩阵可视化</li>
                <li>详细策略概率分析</li>
                <li>公共牌设置和冲突检测</li>
                <li>策略数据导出</li>
            </ul>
            """
        )
    
    def _on_help(self) -> None:
        """处理使用说明对话框。"""
        QMessageBox.information(
            self,
            "使用说明",
            """<h3>策略查看器使用说明</h3>
            
            <h4>1. 加载模型</h4>
            <p>点击 文件 → 加载检查点，选择 .pt 或 .pth 文件。</p>
            
            <h4>2. 浏览游戏树</h4>
            <p>在左侧游戏树中点击节点，查看不同决策点的策略。</p>
            
            <h4>3. 查看手牌策略</h4>
            <p>在中央矩阵中悬停或点击手牌格子，查看详细策略。</p>
            
            <h4>4. 设置公共牌</h4>
            <p>切换到"公共牌设置"标签页，选择翻牌/转牌/河牌。</p>
            
            <h4>5. 导出数据</h4>
            <p>使用 导出 菜单将矩阵图片或策略JSON保存到文件。</p>
            
            <h4>快捷键</h4>
            <ul>
                <li>Ctrl+O: 加载检查点</li>
                <li>Ctrl+Q: 退出</li>
                <li>F5: 刷新显示</li>
                <li>Ctrl+E: 展开所有节点</li>
                <li>Ctrl+Home: 重置到根节点</li>
            </ul>
            """
        )

    # ========================================================================
    # UI事件处理
    # ========================================================================
    
    def _update_action_buttons(self) -> None:
        """更新行动按钮。"""
        # 清除现有按钮
        while self._action_buttons_layout.count():
            item = self._action_buttons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 获取当前节点
        node = self._controller.get_current_node()
        if not node:
            return
        
        # 更新阶段和玩家信息
        stage_names = {
            GameStage.PREFLOP: "翻牌前",
            GameStage.FLOP: "翻牌",
            GameStage.TURN: "转牌",
            GameStage.RIVER: "河牌",
        }
        stage_name = stage_names.get(node.stage, str(node.stage.value))
        
        if node.player >= 0:
            self._stage_label.setText(f"{stage_name} - P{node.player + 1}")
        else:
            self._stage_label.setText(stage_name)
        
        # 获取可用行动（子节点）
        children = node.children
        
        # 行动按钮颜色
        action_colors = {
            ActionType.FOLD: "#e74c3c",      # 红色
            ActionType.CHECK: "#3498db",     # 蓝色
            ActionType.CALL: "#2ecc71",      # 绿色
            ActionType.RAISE: "#f39c12",     # 橙色
        }
        
        # 行动名称
        action_names = {
            ActionType.FOLD: "弃牌",
            ActionType.CHECK: "过牌",
            ActionType.CALL: "跟注",
            ActionType.RAISE: "加注",
            ActionType.RAISE_SMALL: "小加注",
            ActionType.RAISE_BIG: "大加注",
        }
        
        raise_types = (ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG)
        
        for child in children:
            if child.action is not None:
                action = child.action
                action_type = action.action_type
                
                # 获取按钮文本
                if action_type in raise_types:
                    btn_text = f"加注 ${action.amount}"
                else:
                    btn_text = action_names.get(action_type, action_type.value)
                
                # 获取按钮颜色
                color = action_colors.get(action_type, "#95a5a6")
                
                # 创建按钮
                btn = QPushButton(btn_text)
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {color};
                        color: white;
                        border: none;
                        padding: 8px 20px;
                        border-radius: 5px;
                        font-size: 13px;
                        font-weight: bold;
                        min-width: 80px;
                    }}
                    QPushButton:hover {{
                        background-color: {color}dd;
                    }}
                """)
                btn.clicked.connect(lambda checked, n=child: self._on_action_clicked(n))
                self._action_buttons_layout.addWidget(btn)
            elif child.node_type == NodeType.CHANCE:
                # 发牌节点
                stage_names_chance = {
                    GameStage.FLOP: "进入翻牌",
                    GameStage.TURN: "进入转牌",
                    GameStage.RIVER: "进入河牌",
                }
                btn_text = stage_names_chance.get(child.stage, "发牌")
                btn = QPushButton(btn_text)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #9b59b6;
                        color: white;
                        border: none;
                        padding: 8px 20px;
                        border-radius: 5px;
                        font-size: 13px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #8e44ad;
                    }
                """)
                btn.clicked.connect(lambda checked, n=child: self._on_action_clicked(n))
                self._action_buttons_layout.addWidget(btn)
            elif child.node_type == NodeType.PLAYER and child.action is None:
                # 决策节点（没有行动）
                btn = QPushButton("继续")
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #34495e;
                        color: white;
                        border: none;
                        padding: 8px 20px;
                        border-radius: 5px;
                        font-size: 13px;
                    }
                    QPushButton:hover {
                        background-color: #2c3e50;
                    }
                """)
                btn.clicked.connect(lambda checked, n=child: self._on_action_clicked(n))
                self._action_buttons_layout.addWidget(btn)
        
        # 添加弹性空间
        self._action_buttons_layout.addStretch()
    
    def _on_action_clicked(self, node: GameTreeNode) -> None:
        """处理行动按钮点击。
        
        Args:
            node: 目标节点
        """
        # 导航到该节点
        self._controller.navigate_to(node)
        self._game_tree_widget.select_node(node)
        
        # 更新显示
        self._update_node_display(node)
        self._update_action_buttons()
        
        # 如果是发牌节点，切换到公共牌设置
        if node.node_type == NodeType.CHANCE:
            self._right_tab_widget.setCurrentWidget(self._board_selector_widget)
            stage_cards = {
                GameStage.FLOP: 3,
                GameStage.TURN: 4,
                GameStage.RIVER: 5,
            }
            num_cards = stage_cards.get(node.stage, 0)
            current_cards = len(self._controller.get_board_cards())
            if current_cards < num_cards:
                self.statusBar().showMessage(
                    f"请选择公共牌（需要{num_cards}张，当前{current_cards}张）", 5000
                )
    
    def _on_back_clicked(self) -> None:
        """处理返回按钮点击。"""
        node = self._controller.get_current_node()
        if node and node.parent:
            self._controller.navigate_to(node.parent)
            self._game_tree_widget.select_node(node.parent)
            self._update_node_display(node.parent)
            self._update_action_buttons()
            self.statusBar().showMessage("已返回上一步", 2000)
    
    def _on_node_selected(self, node: GameTreeNode) -> None:
        """处理游戏树节点选择事件。
        
        Args:
            node: 被选中的节点
        """
        # 导航到该节点
        self._controller.navigate_to(node)
        
        # 更新路径显示
        self._update_path_display(node)
        
        # 更新行动按钮
        self._update_action_buttons()
        
        # 如果是发牌节点（CHANCE），自动切换到公共牌设置标签页
        if node.node_type == NodeType.CHANCE:
            self._right_tab_widget.setCurrentWidget(self._board_selector_widget)
            # 根据阶段提示需要选择的牌数
            stage_cards = {
                GameStage.FLOP: 3,
                GameStage.TURN: 4,
                GameStage.RIVER: 5,
            }
            num_cards = stage_cards.get(node.stage, 0)
            current_cards = len(self._controller.get_board_cards())
            if current_cards < num_cards:
                self.statusBar().showMessage(
                    f"请选择公共牌（需要{num_cards}张，当前{current_cards}张）", 5000
                )
        else:
            # 显示状态消息
            self.statusBar().showMessage(f"已选择节点: {node.node_id}", 2000)
    
    def _on_hand_hovered(self, hand_label: str) -> None:
        """处理手牌悬停事件。
        
        Args:
            hand_label: 手牌标签
        """
        # 更新状态栏
        strategy = self._controller.get_hand_strategy(hand_label)
        if strategy:
            action = strategy.get_dominant_action()
            if action:
                prob = strategy.action_probabilities.get(action, 0.0)
                self.statusBar().showMessage(
                    f"{hand_label}: 主要行动 {action} ({prob*100:.1f}%)"
                )
            else:
                self.statusBar().showMessage(f"{hand_label}: 无策略数据")
        else:
            self.statusBar().showMessage(f"{hand_label}")
    
    def _on_hand_clicked(self, hand_label: str) -> None:
        """处理手牌点击事件。
        
        Args:
            hand_label: 手牌标签
        """
        # 获取手牌策略
        strategy = self._controller.select_hand(hand_label)
        
        if strategy:
            # 更新策略详情面板
            self._strategy_detail_widget.set_hand_strategy(strategy)
            
            # 切换到策略详情标签页
            self._right_tab_widget.setCurrentWidget(self._strategy_detail_widget)
            
            self.statusBar().showMessage(f"已选择手牌: {hand_label}", 2000)
        else:
            self.statusBar().showMessage(f"手牌 {hand_label} 无策略数据", 2000)
    
    def _on_board_changed(self, cards: List[Card]) -> None:
        """处理公共牌变化事件。
        
        Args:
            cards: 新的公共牌列表
        """
        # 更新控制器中的公共牌
        self._controller.set_board_cards(cards)
        
        # 显示状态消息
        if cards:
            card_str = " ".join(str(c) for c in cards)
            self.statusBar().showMessage(f"公共牌已更新: {card_str}", 2000)
        else:
            self.statusBar().showMessage("公共牌已清空", 2000)

    # ========================================================================
    # 控制器回调处理
    # ========================================================================
    
    def _on_model_loaded_callback(self, metadata: ModelMetadata) -> None:
        """模型加载完成回调。
        
        Args:
            metadata: 模型元数据
        """
        # 获取动作配置并传递给策略详情控件
        # 需求引用: 需求 5.1 - 动态生成与动作数量匹配的概率条
        action_config = self._controller.get_action_config()
        if action_config is not None:
            self._strategy_detail_widget.set_action_config(action_config)
        
        # 更新信息面板
        node = self._controller.get_current_node()
        if node:
            self._update_node_display(node)
    
    def _on_node_changed_callback(self, node: GameTreeNode) -> None:
        """节点变化回调。
        
        Args:
            node: 新的当前节点
        """
        self._update_node_display(node)
        self._update_action_buttons()
    
    def _on_strategy_updated_callback(self, result: StrategyResult) -> None:
        """策略更新回调。
        
        Args:
            result: 策略计算结果
        """
        # 更新手牌矩阵
        strategies = {}
        for label, strategy in result.hand_strategies.items():
            strategies[label] = strategy.action_probabilities
        self._hand_matrix_widget.set_strategies(strategies)
        
        # 更新禁用的手牌
        disabled_hands = set()
        for label, strategy in result.hand_strategies.items():
            if not strategy.combinations:
                disabled_hands.add(label)
        self._hand_matrix_widget.set_disabled_hands(disabled_hands)
        
        # 更新信息面板
        if result.node_state:
            self._info_panel_widget.set_node_state(result.node_state)
    
    def _on_board_changed_callback(self, cards: List[Card]) -> None:
        """公共牌变化回调。
        
        Args:
            cards: 新的公共牌列表
        """
        # 同步公共牌选择器（如果是从控制器触发的变化）
        current_cards = self._board_selector_widget.get_selected_cards()
        if current_cards != cards:
            self._board_selector_widget.set_selected_cards(cards)
    
    def _on_value_updated_callback(self, result) -> None:
        """价值更新回调。
        
        Args:
            result: 价值计算结果
        """
        if result is not None:
            # 检查是否有公共牌（翻后阶段需要）
            board_cards = self._controller.get_board_cards()
            has_board_cards = len(board_cards) > 0
            self._value_heatmap_widget.set_value_result(result, has_board_cards)
        else:
            self._value_heatmap_widget.set_unavailable()
    
    def _on_value_hand_hovered(self, hand_label: str) -> None:
        """处理价值热图手牌悬停事件。
        
        Args:
            hand_label: 手牌标签
        """
        hand_value = self._controller.get_hand_value(hand_label)
        if hand_value:
            if hand_value.is_blocked:
                self.statusBar().showMessage(f"{hand_label}: 被公共牌阻挡")
            else:
                self.statusBar().showMessage(
                    f"{hand_label}: 价值估计 = {hand_value.average_value:.4f}"
                )
        else:
            self.statusBar().showMessage(f"{hand_label}: 价值网络不可用")
    
    def _on_value_hand_clicked(self, hand_label: str) -> None:
        """处理价值热图手牌点击事件。
        
        Args:
            hand_label: 手牌标签
        """
        hand_value = self._controller.get_hand_value(hand_label)
        if hand_value and hand_value.combination_values:
            # 显示详细的组合价值
            details = []
            for combo, value in hand_value.combination_values.items():
                details.append(f"{combo}: {value:.4f}")
            
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                f"{hand_label} 价值详情",
                f"平均价值: {hand_value.average_value:.4f}\n\n" +
                "各组合价值:\n" + "\n".join(details[:12])  # 最多显示12个
            )
    
    def _on_regret_updated_callback(self, result) -> None:
        """遗憾值更新回调。
        
        Args:
            result: 遗憾值计算结果
        """
        if result is not None:
            # 检查是否有公共牌（翻后阶段需要）
            board_cards = self._controller.get_board_cards()
            has_board_cards = len(board_cards) > 0
            self._regret_heatmap_widget.set_regret_result(result, has_board_cards)
        else:
            self._regret_heatmap_widget.set_unavailable()
    
    def _on_regret_hand_hovered(self, hand_label: str) -> None:
        """处理遗憾值热图手牌悬停事件。
        
        Args:
            hand_label: 手牌标签
        """
        hand_regret = self._controller.get_hand_regret(hand_label)
        if hand_regret:
            if hand_regret.is_blocked:
                self.statusBar().showMessage(f"{hand_label}: 被公共牌阻挡")
            else:
                # 显示各动作的遗憾值
                regrets_str = " | ".join(
                    f"{name}:{val:+.2f}" 
                    for name, val in hand_regret.average_regrets.items()
                )
                self.statusBar().showMessage(f"{hand_label}: {regrets_str}")
        else:
            self.statusBar().showMessage(f"{hand_label}: 遗憾网络不可用")
    
    def _on_regret_hand_clicked(self, hand_label: str) -> None:
        """处理遗憾值热图手牌点击事件。
        
        Args:
            hand_label: 手牌标签
        """
        hand_regret = self._controller.get_hand_regret(hand_label)
        if hand_regret and hand_regret.combination_regrets:
            # 显示详细的组合遗憾值
            details = []
            for combo, regrets in list(hand_regret.combination_regrets.items())[:6]:
                regret_str = " ".join(f"{k}:{v:+.2f}" for k, v in regrets.items())
                details.append(f"{combo}: {regret_str}")
            
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                f"{hand_label} 遗憾值详情",
                f"平均遗憾值:\n" +
                "\n".join(f"  {k}: {v:+.4f}" for k, v in hand_regret.average_regrets.items()) +
                f"\n\n总正遗憾值: {hand_regret.average_total_regret:.4f}\n\n" +
                "各组合遗憾值:\n" + "\n".join(details)
            )
    
    def _update_path_display(self, node: GameTreeNode) -> None:
        """更新路径显示。
        
        Args:
            node: 当前节点
        """
        # 获取路径
        path = node.get_path_to_root()
        
        # 构建路径文本
        path_parts = []
        action_names = {
            ActionType.FOLD: "弃牌",
            ActionType.CHECK: "过牌",
            ActionType.CALL: "跟注",
            ActionType.RAISE: "加注",
            ActionType.RAISE_SMALL: "小加注",
            ActionType.RAISE_BIG: "大加注",
        }
        
        raise_types = (ActionType.RAISE, ActionType.RAISE_SMALL, ActionType.RAISE_BIG)
        
        for i, n in enumerate(path):
            if n.node_type == NodeType.ROOT:
                path_parts.append("游戏开始")
            elif n.action is not None:
                action = n.action
                if action.action_type in raise_types:
                    path_parts.append(f"加注${action.amount}")
                else:
                    path_parts.append(action_names.get(action.action_type, action.action_type.value))
            elif n.node_type == NodeType.CHANCE:
                stage_names = {
                    GameStage.FLOP: "翻牌",
                    GameStage.TURN: "转牌",
                    GameStage.RIVER: "河牌",
                }
                path_parts.append(stage_names.get(n.stage, n.stage.value))
        
        self._path_label.setText(" → ".join(path_parts))
    
    def _update_node_display(self, node: GameTreeNode) -> None:
        """更新节点相关的显示。
        
        Args:
            node: 当前节点
        """
        # 更新信息面板
        state = NodeState.from_game_tree_node(node)
        self._info_panel_widget.set_node_state(state)
        
        # 更新路径显示
        self._update_path_display(node)
        
        # 清除手牌选择
        self._hand_matrix_widget.clear_selection()
        self._strategy_detail_widget.clear()

    # ========================================================================
    # 公共方法
    # ========================================================================
    
    def get_controller(self) -> StrategyViewerController:
        """获取策略查看器控制器。
        
        Returns:
            控制器实例
        """
        return self._controller
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点文件。
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            是否加载成功
        """
        try:
            metadata = self._controller.load_model(checkpoint_path)
            self._model_status_label.setText(
                f"已加载: {Path(checkpoint_path).name}"
            )
            self._update_menu_state()
            return True
        except Exception as e:
            QMessageBox.critical(
                self,
                "加载失败",
                f"无法加载检查点文件:\n{str(e)}"
            )
            return False
    
    def set_board_cards(self, cards: List[Card]) -> None:
        """设置公共牌。
        
        Args:
            cards: 公共牌列表
        """
        self._board_selector_widget.set_selected_cards(cards)
        self._controller.set_board_cards(cards)
    
    def navigate_to_node(self, node_id: str) -> bool:
        """导航到指定节点。
        
        Args:
            node_id: 节点ID
            
        Returns:
            是否导航成功
        """
        success = self._controller.navigate_to_by_id(node_id)
        if success:
            self._game_tree_widget.select_node_by_id(node_id)
        return success
    
    def select_hand(self, hand_label: str) -> None:
        """选择指定的手牌。
        
        Args:
            hand_label: 手牌标签
        """
        self._hand_matrix_widget.select_hand(hand_label)
        self._on_hand_clicked(hand_label)


def create_main_window(checkpoint_path: Optional[str] = None) -> MainWindow:
    """创建并初始化主窗口的工厂函数。
    
    Args:
        checkpoint_path: 可选的检查点文件路径，如果提供则自动加载
        
    Returns:
        初始化好的主窗口实例
    """
    window = MainWindow()
    
    if checkpoint_path:
        window.load_checkpoint(checkpoint_path)
    
    return window

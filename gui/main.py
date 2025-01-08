import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow,QHBoxLayout, QLabel,
 QStackedWidget,
)
from docker.errors import DockerException
import docker
import webbrowser
from PyQt5.QtGui import QIcon, QFont,QPixmap
from PyQt5.QtCore import Qt
import psutil
import yaml
from env import AGENTS, NEXUSAI_API_BASE_URL
from PyQt5.QtWidgets import (
    QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QFormLayout, QLineEdit, QPushButton, QFileDialog, QMessageBox, QDialog, QTextEdit
)
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal,QObject
from Main import jarvis_brain,recognize_speech
import queue
import threading
from env import NEXUSAI_API_KEY


class EditAgentFormDialog(QDialog):
    """Popup dialog for editing an agent."""
    def __init__(self, agent_data=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Agent")
        self.setGeometry(100, 100, 400, 400)

        self.agent_data = agent_data or {}

        # Set the dialog's layout
        layout = QFormLayout()

        # Input fields
        self.agent_name_input = QLineEdit(self.agent_data.get("name", ""))
        layout.addRow("Agent Name:", self.agent_name_input)

        self.agent_description_input = QTextEdit(self.agent_data.get("description", ""))
        layout.addRow("Agent Description:", self.agent_description_input)

        self.agent_json_path = QLineEdit(self.agent_data.get("json_file", ""))
        self.agent_json_path.setReadOnly(True)
        layout.addRow("JSON Path:", self.agent_json_path)

        upload_json_btn = QPushButton("Upload JSON File")
        upload_json_btn.clicked.connect(self.upload_json_file)
        layout.addRow(upload_json_btn)

        self.save_btn = QPushButton("Save Changes")
        self.save_btn.setStyleSheet("""
                            * {
                                background-color: green;
                                color: white;
                                font-weight: bold;
                                padding: 4px;
                            }
                        """)
        self.save_btn.clicked.connect(self.save_changes)
        layout.addRow(self.save_btn)

        self.setLayout(layout)

    def upload_json_file(self):
        """Upload and select a JSON file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Upload JSON File", "", "JSON Files (*.json)")
        if file_path:
            self.agent_json_path.setText(file_path)

    def save_changes(self):
        """Save updated agent data."""
        updated_data = {
            "name": self.agent_name_input.text(),
            "description": self.agent_description_input.toPlainText(),
            "json_file": self.agent_json_path.text()
        }

        if not updated_data["name"].strip():
            QMessageBox.critical(self, "Error", "Agent name is required.")
            return
        if not updated_data["description"].strip():
            QMessageBox.critical(self, "Error", "Agent description is required.")
            return
        if not updated_data["json_file"].strip():
            QMessageBox.critical(self, "Error", "You must upload a JSON file.")
            return

        try:
            agent_path = os.path.join(str(AGENTS), "agents.yml")
            # Load the data from the YAML file
            if os.path.exists(agent_path):
                with open(agent_path, 'r') as file:
                    agents = yaml.safe_load(file) or []
            else:
                agents = []

            # Update the selected agent's data
            for idx, agent in enumerate(agents):
                if agent.get("name") == self.agent_data.get("name"):  # Match by name
                    agents[idx] = updated_data
                    break

            # Write the updated agents list back to YAML
            with open(agent_path, 'w') as file:
                yaml.dump(agents, file)

            QMessageBox.information(self, "Success", "Changes saved successfully.")
            self.accept()  # Close dialog
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save changes: {e}")



class AddAgentFormDialog(QDialog):
    """Popup dialog for creating a new agent."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Agent")
        self.setGeometry(100, 100, 400, 400)

        # Set the dialog's layout
        layout = QFormLayout()

        # Input fields
        self.agent_name_input = QLineEdit()
        self.agent_name_input.setPlaceholderText("Enter Agent Name")
        layout.addRow("Agent Name:", self.agent_name_input)

        self.agent_description_input = QTextEdit()
        self.agent_description_input.setPlaceholderText("Write agent description (20-150 words)")
        layout.addRow("Agent Description:", self.agent_description_input)

        self.json_file_path = QLineEdit()
        self.json_file_path.setReadOnly(True)  # Prevent direct typing
        layout.addRow("JSON File:", self.json_file_path)

        upload_json_btn = QPushButton("Upload JSON File")
        upload_json_btn.clicked.connect(self.upload_json_file)
        layout.addRow(upload_json_btn)

        self.save_btn = QPushButton("Save Agent")
        self.save_btn.setStyleSheet("""
                            * {
                                background-color: green;
                                color: white;
                                font-weight: bold;
                                padding: 4px;
                            }
                        """)
        self.save_btn.clicked.connect(self.save_agent)
        layout.addRow(self.save_btn)

        self.setLayout(layout)

    def upload_json_file(self):
        """Upload and select a JSON file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Upload JSON File", "", "JSON Files (*.json)")
        if file_path:
            self.json_file_path.setText(file_path)

    def save_agent(self):
        """Save agent details to YAML."""
        name = self.agent_name_input.text()
        description = self.agent_description_input.toPlainText()
        json_file = self.json_file_path.text()

        if not name.strip():
            QMessageBox.critical(self, "Error", "Agent name is required.")
            return
        if not description.strip():
            QMessageBox.critical(self, "Error", "Agent description is required.")
            return

        if not json_file.strip():
            QMessageBox.critical(self, "Error", "You must upload a JSON file.")
            return

        # Save data to the YAML file
        try:
            if not os.path.exists(str(AGENTS)):
                os.makedirs(str(AGENTS),exist_ok=True)
            agent_path  = os.path.join(str(AGENTS), "agents.yml")

            # Check if file exists
            if os.path.exists(agent_path):
                with open(agent_path, 'r') as file:
                    agents = yaml.safe_load(file) or []
            else:
                agents = []

            # Append the new agent
            agents.append({
                "name": name,
                "description": description,
                "path": json_file,
            })

            # Save back to file
            with open(agent_path, 'w') as file:
                yaml.dump(agents, file)

            QMessageBox.information(self, "Success", "Agent saved successfully.")
            self.accept()  # Close dialog on success
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save agent: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.timer = None
        self.network_data = []
        self.ram_data = []
        self.gpu_data = []
        # Window settings
        self.setWindowTitle("SmartHome Nexus")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        # Variables to store settings
        self.agent_path = os.path.join(str(AGENTS), "agents.yml")
        # Main layout
        main_layout = QHBoxLayout()

        # Create navigation bar
        self.nav_bar = self.create_nav_bar()
        main_layout.addWidget(self.nav_bar)

        # Create main content area (central widget)
        self.content_area = QStackedWidget()
        main_layout.addWidget(self.content_area)

        # Add sample pages to content area
        self.content_area.addWidget(self.create_home_page())
        self.content_area.addWidget(self.create_analysis_page())
        self.content_area.addWidget(self.create_reports_page())
        self.content_area.addWidget(self.create_settings_page())
        self.content_area.addWidget(self.create_agent_page())  # Add Agent Page
        self.content_area.addWidget(self.create_docker_page())
        # Set up the main widget and layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def show_error_message(self, title, message):
        """Displays an error message box."""
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle(title)
        error_box.setText(message)
        error_box.setStandardButtons(QMessageBox.Ok)
        error_box.exec_()

    def show_success_message(self, title, message):
        """Displays a success message box."""
        success_box = QMessageBox()
        success_box.setIcon(QMessageBox.Information)  # Icon for success/information
        success_box.setWindowTitle(title)
        success_box.setText(message)
        success_box.setStandardButtons(QMessageBox.Ok)
        success_box.exec_()

    def create_nav_bar(self):
        # Navigation container
        nav_widget = QWidget()
        nav_widget.setFixedWidth(200)
        nav_layout = QVBoxLayout()
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(10)
        nav_widget.setStyleSheet("background-color: #2e2e2e;")

        # Title / Logo at top
        title = QLabel("SmartHome Nexus")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(title)

        # Navigation buttons
        def create_nav_button(name, icon_path, page_index):
            from env import JARVIS_DIR
            path=os.path.join(JARVIS_DIR,"gui","icons",icon_path)
            btn = QPushButton(name)
            btn.setIcon(QIcon(path))  # Placeholder for icons
            btn.clicked.connect(lambda: self.content_area.setCurrentIndex(page_index))
            btn.setStyleSheet("color: white; background-color: #3e3e3e; padding: 10px;")
            return btn

        btn_home = create_nav_button("Home", "home.svg", 0)
        btn_analysis = create_nav_button("Data Analysis", "bar-chart.svg", 1)
        btn_reports = create_nav_button("Reports", "file-text.svg", 2)
        btn_settings = create_nav_button("Settings", "settings.svg", 3)
        btn_agent = create_nav_button("Agent", "agent.svg", 4) # New Agent Button
        docker_btn=create_nav_button("Docker","docker.svg",5)


        nav_layout.addWidget(btn_home)
        nav_layout.addWidget(btn_analysis)
        nav_layout.addWidget(btn_reports)
        nav_layout.addWidget(btn_agent)  # Add Agent button
        nav_layout.addWidget(btn_settings)
        nav_layout.addWidget(docker_btn)

        # Add spacer at bottom
        nav_layout.addStretch()

        nav_widget.setLayout(nav_layout)
        return nav_widget

    def create_home_page(self):
        page = QWidget()
        self.home_layout = QVBoxLayout()

        # Title
        title = QLabel("Home")
        title.setFont(QFont("Arial", 24))
        title.setAlignment(Qt.AlignCenter)
        self.home_layout.addWidget(title)

        # Configurations Section
        self.config_label = QLabel()
        self.update_home_page_configurations()
        # Set layout
        #self.log_label = QLabel("Press Start to initialize J.A.R.V.I.S.")
        #self.home_layout.addWidget(self.log_label)

        #self.start_button = QPushButton("Start J.A.R.V.I.S")
        #self.start_button.clicked.connect(self.start_service)
        #self.home_layout.addWidget(self.start_button)
        #self.stop_button = QPushButton("Stop J.A.R.V.I.S")
        #self.stop_button.clicked.connect(self.stop_service)
        #self.stop_button.setEnabled(False)
        #self.home_layout.addWidget(self.stop_button)
        self.home_layout.addWidget(self.config_label)

        # Add spacer
        self.home_layout.addStretch()
        page.setLayout(self.home_layout)
        return page

    def update_home_page_configurations(self):
        """Update the configurations displayed on the home page."""
        from env import CACHE_DIR
        config_dir = os.path.join(CACHE_DIR, "config")
        config_path = os.path.join(config_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path,"r") as f:
                data=dict(json.load(f))
                config_text="""
                <h3>Configuration<h3>
                """

            config_text+=f"""
            <pre><b>Nexus AI:{"True" if data['nexusai']['api_key'] else "False"}</b></pre>
            """
            config_text += f"""
            <pre><b>HuggingFace:{"True" if data['hf']['api-key'] else "False"}</b></pre>
            """
            config_text += f"""
                        <pre><b>Groq Cloud:{"True" if data['groq'] else "False"}</b></pre>
                        """
            config_text += f"""
                        <pre><b>Home Assistant:{"True" if data['automation']['home-assistant']['base_url'] else "False"}</b></pre>
                        """

        else:
            config_text="""
            <h3>Selected Configurations:</h3>
            <p>No Configuration set</p>
            """
        self.config_label.setText(config_text)
        self.config_label.setStyleSheet("font-size: 14px;")

    def create_analysis_page(self):
        page = QWidget()
        layout = QVBoxLayout()

        # Storage Information Table
        storage_label = QLabel("Storage Information:")
        storage_label.setFont(QFont("Arial", 14))
        layout.addWidget(storage_label)

        storage_table = QTableWidget()
        storage_table.setColumnCount(3)
        storage_table.setHorizontalHeaderLabels(["Drive", "Total Storage (GB)", "Used Storage (GB)"])
        storage_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        storage_table.setStyleSheet("""
        {
            border:none;
            background-color:black;
            color:white;
        }
        """)

        partitions = psutil.disk_partitions()
        storage_table.setRowCount(len(partitions))
        for i, partition in enumerate(partitions):
            usage = psutil.disk_usage(partition.mountpoint)
            storage_table.setItem(i, 0, QTableWidgetItem(partition.device))
            storage_table.setItem(i, 1, QTableWidgetItem(f"{usage.total / (1024 ** 3):.2f}"))
            storage_table.setItem(i, 2, QTableWidgetItem(f"{usage.used / (1024 ** 3):.2f}"))

        layout.addWidget(storage_table)

        # Network Graph
        network_label = QLabel("Network Usage:")
        network_label.setFont(QFont("Arial", 14))
        layout.addWidget(network_label)

        self.network_graph = pg.PlotWidget()
        self.network_graph.setBackground('w')
        self.network_graph.setTitle("Network Usage", color="k", size="12pt")
        self.network_graph.setLabel('left', 'Speed (KB/s)', color="k")
        self.network_graph.setLabel('bottom', 'Time (s)', color="k")
        self.download_curve = self.network_graph.plot(pen=pg.mkPen('r', width=2), name="Download")
        self.upload_curve = self.network_graph.plot(pen=pg.mkPen('g', width=2), name="Upload")
        layout.addWidget(self.network_graph)

        # RAM Usage Graph
        ram_label = QLabel("RAM Usage:")
        ram_label.setFont(QFont("Arial", 14))
        layout.addWidget(ram_label)

        self.ram_graph = pg.PlotWidget()
        self.ram_graph.setBackground('w')
        self.ram_graph.setTitle("RAM Usage", color="k", size="12pt")
        self.ram_graph.setLabel('left', 'Usage (GB)', color="k")
        self.ram_graph.setLabel('bottom', 'Time (s)', color="k")
        self.ram_curve = self.ram_graph.plot(pen=pg.mkPen('b', width=2), name="RAM")
        layout.addWidget(self.ram_graph)

        # GPU Usage Graph (if available)
        try:
            import GPUtil
            gpu_label = QLabel("GPU Usage:")
            gpu_label.setFont(QFont("Arial", 14))
            layout.addWidget(gpu_label)

            self.gpu_graph = pg.PlotWidget()
            self.gpu_graph.setBackground('w')
            self.gpu_graph.setTitle("GPU Usage", color="k", size="12pt")
            self.gpu_graph.setLabel('left', 'Usage (%)', color="k")
            self.gpu_graph.setLabel('bottom', 'Time (s)', color="k")
            self.gpu_curve = self.gpu_graph.plot(pen=pg.mkPen('m', width=2), name="GPU")
            layout.addWidget(self.gpu_graph)
        except ImportError:
            pass

        # Set timer for updating graphs
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_graphs)
        self.timer.start(1000)  # Update every second

        # Add spacer at bottom
        layout.addStretch()

        page.setLayout(layout)
        return page

    def update_graphs(self):
        # Update Network Data
        net_io = psutil.net_io_counters()
        download_speed = net_io.bytes_recv / 1024  # KB/s
        upload_speed = net_io.bytes_sent / 1024  # KB/s
        self.network_data.append((download_speed, upload_speed))
        if len(self.network_data) > 100:
            self.network_data.pop(0)

        download_speeds = [data[0] for data in self.network_data]
        upload_speeds = [data[1] for data in self.network_data]
        self.download_curve.setData(download_speeds)
        self.upload_curve.setData(upload_speeds)

        # Update RAM Data
        ram_usage = psutil.virtual_memory().used / (1024 ** 3)  # GB
        self.ram_data.append(ram_usage)
        if len(self.ram_data) > 100:
            self.ram_data.pop(0)

        self.ram_curve.setData(self.ram_data)

        # Update GPU Data (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_usage = gpus[0].load * 100  # Convert to percentage
                self.gpu_data.append(gpu_usage)
                if len(self.gpu_data) > 100:
                    self.gpu_data.pop(0)

                self.gpu_curve.setData(self.gpu_data)
        except ImportError:
            pass

    def create_reports_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        label = QLabel("Logs Viewer")
        label.setFont(QFont("Arial", 24))
        layout.addWidget(label)

        # Logs Table
        self.logs_table = QTableWidget()
        self.logs_table.setColumnCount(2)
        self.logs_table.setHorizontalHeaderLabels(["Type", "Message"])
        self.logs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.logs_table)

        # Load logs button
        load_logs_btn = QPushButton("Load Logs")
        load_logs_btn.clicked.connect(self.load_logs)
        layout.addWidget(load_logs_btn)

        # Download logs button
        download_logs_btn = QPushButton("Download Logs as JSON")
        download_logs_btn.clicked.connect(self.download_logs)
        layout.addWidget(download_logs_btn)

        # Spacer at the bottom
        layout.addStretch()
        page.setLayout(layout)
        return page

    def load_logs(self):
        log_dir = "logs"
        if not os.path.exists(log_dir):
            print("Logs directory not found.")
            return

        self.logs_table.setRowCount(0)  # Clear existing rows

        for log_file in os.listdir(log_dir):
            file_path = os.path.join(log_dir, log_file)
            if os.path.isfile(file_path) and log_file.endswith(".log"):
                with open(file_path, "r") as file:
                    for line in file.readlines():
                        log_parts = line.split(" ", 1)  # Split into level and message
                        if len(log_parts) == 2:
                            level, message = log_parts
                            row_pos = self.logs_table.rowCount()
                            self.logs_table.insertRow(row_pos)
                            self.logs_table.setItem(row_pos, 0, QTableWidgetItem(level.strip()))
                            self.logs_table.setItem(row_pos, 1, QTableWidgetItem(message.strip()))

    def download_logs(self):
        logs = []
        for row in range(self.logs_table.rowCount()):
            log_type = self.logs_table.item(row, 0).text()
            message = self.logs_table.item(row, 1).text()
            logs.append({"type": log_type, "message": message})

        # Save JSON file
        file_dialog = QFileDialog()
        save_path, _ = file_dialog.getSaveFileName(self, "Save Logs as JSON", "", "JSON Files (*.json)")
        if save_path:
            with open(save_path, "w") as json_file:
                json.dump(logs, json_file, indent=4)
            print(f"Logs saved to {save_path}")

    def create_agent_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Display existing agents
        title = QLabel("Agents")
        title.setFont(QFont("Arial", 16))
        layout.addWidget(title)

        # Table to display agents
        self.agent_table = QTableWidget()
        self.agent_table.setColumnCount(4)  # Added 4 columns now: Name, Description, Edit, Delete
        self.agent_table.setHorizontalHeaderLabels(["Agent Name", "Agent Description", "Edit", "Delete"])
        self.agent_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.load_agents()
        layout.addWidget(self.agent_table)
        # Styling header
        self.agent_table.setStyleSheet("""
                QHeaderView::section {
                    background-color: black;
                    color: white;
                    font-weight: bold;
                    padding: 4px;
                }
            """)

        # Populate agent table with initial data
        self.load_agents()  # Load initial agents data
        layout.addWidget(self.agent_table)
        # Button to open Create Agent dialog
        create_agent_btn = QPushButton("Create Agent")
        create_agent_btn.setStyleSheet("""
                                * {
                                    background-color: green;
                                    color: white;
                                    font-weight: bold;
                                    padding: 6px;
                                }
                            """)
        create_agent_btn.clicked.connect(self.open_create_agent_dialog)
        layout.addWidget(create_agent_btn)


        # Spacer at bottom
        layout.addStretch()
        page.setLayout(layout)
        return page

    def open_create_agent_dialog(self):
        """Open the modal dialog to create a new agent."""
        dialog = AddAgentFormDialog(self)
        if dialog.exec_():
            self.load_agents()  # Refresh the agent table after successful submission

    def load_agents(self):
        """Load agents data into the QTableWidget dynamically."""
        import yaml
        try:
            if os.path.exists(self.agent_path):
                with open(self.agent_path, "r") as file:
                    agents = yaml.safe_load(file) or []
            else:
                agents = []

            # Clear table before repopulating
            self.agent_table.setRowCount(0)

            # Dynamically add rows from YAML
            self.agent_table.setRowCount(len(agents))
            for row, agent in enumerate(agents):
                self.agent_table.setItem(row, 0, QTableWidgetItem(agent.get("name")))
                self.agent_table.setItem(row, 1, QTableWidgetItem(agent.get("description")))
                # Edit Button
                edit_btn = QPushButton("Edit")
                edit_btn.clicked.connect(lambda checked, ag=agent: self.edit_agent(ag))
                self.agent_table.setCellWidget(row, 2, edit_btn)

                # Delete Button
                del_btn = QPushButton("Delete")
                del_btn.setStyleSheet("""
                                            background-color: red;
                                            color: white;
                                            font-weight: bold;
                                        """)
                del_btn.clicked.connect(lambda checked, ag=agent: self.delete_agent(ag))
                self.agent_table.setCellWidget(row, 3, del_btn)
        except Exception as e:
            self.show_error_message("Failed to Load Agents", f"Error: {str(e)}")

    def edit_agent(self, agent):
        """Open a dialog to edit the agent."""
        dialog = EditAgentFormDialog(agent, self)
        if dialog.exec_():
            self.load_agents()

    def delete_agent(self, agent):
        """Delete agent from YAML."""
        try:
            with open(self.agent_path, 'r') as file:
                agents = yaml.safe_load(file) or []
            # Remove the selected agent
            agents = [ag for ag in agents if ag.get("name") != agent.get("name")]
            if len(agents)!=0:
                # Write the new list back to the YAML file
                with open(self.agent_path, 'w') as file:
                    yaml.dump(agents, file)
            else:
                os.remove(self.agent_path)

            QMessageBox.information(self, "Success", "Agent deleted successfully.")
            self.load_agents()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete agent: {e}")
    def create_cam(self):
        """

        :return:
        """
    def create_docker_page(self):
        """
        :return: QWidget
        """
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Title for the container section
        title = QLabel("Docker Containers")
        title.setFont(QFont("Arial", 16))
        layout.addWidget(title)

        try:
            # Establish connection with Docker
            # Create a table for displaying agents
            self.agent_table = QTableWidget()
            self.agent_table.setColumnCount(5)  # Added 5 columns: ID, Name, Status, Actions, Open URL
            self.agent_table.setHorizontalHeaderLabels(["ID", "Name", "Status", "Actions", "Open URL"])
            self.agent_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

            # Add the table to the UI
            layout.addWidget(self.agent_table)

            # Styling for table header
            self.agent_table.setStyleSheet("""
                            QHeaderView::section {
                                background-color: black;
                                color: white;
                                font-weight: bold;
                                padding: 4px;
                            }
                        """)

            # Load initial agent data from running containers
            self.load_container()

            # Handle exception and connection issues
        except DockerException as e:
            QMessageBox.critical(self, "Docker Error", f"Error connecting to Docker: {e}")

        page.setLayout(layout)
        return page

    def is_docker_installed(self):
        """
        Check if Docker is installed by attempting to create a client.
        """
        try:
            print("Checking the Docker.")
            client = docker.from_env()
            client.ping()
            return True
        except DockerException:
            return False
    def load_container(self):
        """
        Load agents (containers) from Docker and populate the table dynamically.
        """

        try:
            if self.is_docker_installed():
                client = docker.from_env()
                containers = client.containers.list(all=True)

                # Clear the table
                self.agent_table.setRowCount(0)

                # Populate table with the containers' information
                for idx, container in enumerate(containers):
                    self.agent_table.insertRow(idx)

                    # Set container information into the table rows
                    self.agent_table.setItem(idx, 0, QTableWidgetItem(container.short_id))  # Container ID
                    self.agent_table.item(idx, 0).setFlags(Qt.ItemIsEnabled)
                    self.agent_table.setItem(idx, 1, QTableWidgetItem(container.name))  # Container Name
                    status = container.status
                    self.agent_table.item(idx, 1).setFlags(Qt.ItemIsEnabled)
                    self.agent_table.setItem(idx, 2, QTableWidgetItem(status))  # Container Status
                    self.agent_table.item(idx, 2).setFlags(Qt.ItemIsEnabled)
                    # Create dynamic Start/Stop button based on the container's status
                    action_button = QPushButton("Stop" if container.status == "running" else "Start")
                    action_button.clicked.connect(lambda _, c=container: self.toggle_container(c))
                    self.agent_table.setCellWidget(idx, 3, action_button)

                    # Create a clickable "Open URL" button if the container has ports exposed
                    url_button = QPushButton("Open URL")
                    if container.attrs['NetworkSettings']['Ports'] and '80/tcp' in container.attrs['NetworkSettings']['Ports']:
                        url_button.setEnabled(True)
                        url_button.clicked.connect(lambda _, url=container.attrs['NetworkSettings']['Ports']['80/tcp'][0]['HostPort']:
                                                   self.open_url(f"http://localhost:{url}"))
                    else:
                        url_button.setEnabled(False)

                    self.agent_table.setCellWidget(idx, 4, url_button)

        except DockerException as e:
            QMessageBox.critical(self, "Error", f"Could not fetch container information: {e}")

    def toggle_container(self, container):
        """
        Start or stop a container depending on its current state.
        """
        try:
            if container.status == "running":
                container.stop()
                QMessageBox.information(self, "Container Action", f"Stopping container {container.name}")
            else:
                container.start()
                QMessageBox.information(self, "Container Action", f"Starting container {container.name}")

            # Refresh table to reflect the change
            self.load_container()
        except DockerException as e:
            QMessageBox.critical(self, "Docker Error", f"Could not perform action: {e}")

    def open_url(self, url):
        """
        Open a given URL in the system's default web browser.
        """
        try:
            webbrowser.open(url)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not open URL: {e}")


    def save_to_env(self, key, value):
        from dotenv import set_key,load_dotenv
        from env import JARVIS_DIR
        load_dotenv()
        dotenv_path = os.path.join(JARVIS_DIR,".env")
        set_key(dotenv_path, key, value)

    def toggle_visibility(self, line_edit, toggle_button):
            if line_edit.echoMode() == QLineEdit.Password:
                line_edit.setEchoMode(QLineEdit.Normal)
                toggle_button.setStyleSheet("background-color: green; color: white;")
                toggle_button.setText("Hide")
            else:
                line_edit.setEchoMode(QLineEdit.Password)
                toggle_button.setStyleSheet("background-color: green; color: white;")
                toggle_button.setText("Show")

    def create_settings_page(self):
            page = QWidget()
            layout = QVBoxLayout()
            layout.setContentsMargins(20, 20, 20, 20)

            # ChatGPT Settings
            chatgpt_group = QGroupBox("Nexus AI Settings")
            chatgpt_layout = QFormLayout()
            chatgpt_url_input = QLineEdit(os.getenv("NEXUSAI_BASE_URL", ""))
            chatgpt_layout.addRow("Nexus AI Base URL:", chatgpt_url_input)

            chatgpt_api_key_input = QLineEdit(os.getenv("NEXUSAI_API_KEY", ""))
            chatgpt_api_key_input.setEchoMode(QLineEdit.Password)
            chatgpt_api_key_layout = QHBoxLayout()
            chatgpt_api_key_layout.addWidget(chatgpt_api_key_input)

            chatgpt_toggle_btn = QPushButton("Show")
            chatgpt_toggle_btn.clicked.connect(
                lambda: self.toggle_visibility(chatgpt_api_key_input, chatgpt_toggle_btn))
            chatgpt_api_key_layout.addWidget(chatgpt_toggle_btn)

            chatgpt_layout.addRow("Nexus AI API Key:", chatgpt_api_key_layout)
            chatgpt_group.setLayout(chatgpt_layout)
            layout.addWidget(chatgpt_group)

            # Home Assistant Settings
            ha_group = QGroupBox("Home Assistant Settings")
            ha_layout = QFormLayout()
            ha_url_input = QLineEdit(os.getenv("HA_BASE_URL", ""))
            ha_layout.addRow("Home Assistant URL:", ha_url_input)

            ha_api_key_input = QLineEdit(os.getenv("HA_API_KEY", ""))
            ha_api_key_input.setEchoMode(QLineEdit.Password)
            ha_api_key_layout = QHBoxLayout()
            ha_api_key_layout.addWidget(ha_api_key_input)

            ha_toggle_btn = QPushButton("Show")
            ha_toggle_btn.clicked.connect(lambda: self.toggle_visibility(ha_api_key_input, ha_toggle_btn))
            ha_api_key_layout.addWidget(ha_toggle_btn)

            ha_layout.addRow("Home Assistant API Key:", ha_api_key_layout)
            ha_group.setLayout(ha_layout)
            layout.addWidget(ha_group)

            # User Information
            user_info_group = QGroupBox("User Information")
            user_info_layout = QFormLayout()
            user_name_input = QLineEdit(os.getenv("USER_NAME", ""))
            user_info_layout.addRow("Name:", user_name_input)

            # Photo upload
            photo_label = QLabel()
            photo_path = os.getenv("USER_PHOTO", "")
            if photo_path:
                pixmap = QPixmap(photo_path)
                photo_label.setPixmap(pixmap.scaled(100, 100))
            user_info_layout.addRow("Photo:", photo_label)

            upload_photo_btn = QPushButton("Upload Photo")
            upload_photo_btn.setStyleSheet("background-color: green; color: white;")
            upload_photo_btn.clicked.connect(lambda: self.upload_photo(photo_label))
            user_info_layout.addRow(upload_photo_btn)

            # Audio file upload
            audio_file_input = QLineEdit(os.getenv("USER_AUDIO_FILE", ""))
            audio_file_input.setPlaceholderText("Path to audio file")
            audio_upload_btn = QPushButton("Upload Audio File")
            audio_upload_btn.setStyleSheet("background-color: green; color: white;")
            audio_upload_btn.clicked.connect(lambda: self.upload_audio(audio_file_input))
            user_info_layout.addRow("Audio File:", audio_file_input)
            user_info_layout.addRow(audio_upload_btn)

            user_info_group.setLayout(user_info_layout)
            layout.addWidget(user_info_group)

            # HF Token
            hf_group = QGroupBox("Hugging Face Token")
            hf_layout = QFormLayout()
            hf_token_input = QLineEdit(os.getenv("HF_TOKEN", ""))
            hf_token_input.setEchoMode(QLineEdit.Password)
            hf_token_layout = QHBoxLayout()
            hf_token_layout.addWidget(hf_token_input)

            hf_toggle_btn = QPushButton("Show")
            hf_toggle_btn.clicked.connect(lambda: self.toggle_visibility(hf_token_input, hf_toggle_btn))
            hf_token_layout.addWidget(hf_toggle_btn)

            hf_layout.addRow("HF Token:", hf_token_layout)
            hf_group.setLayout(hf_layout)
            layout.addWidget(hf_group)

            # GROQ API Key
            groq_group = QGroupBox("GROQ API Settings")
            groq_layout = QFormLayout()
            groq_api_key_input = QLineEdit(os.getenv("GROQ_API_KEY", ""))
            groq_api_key_input.setEchoMode(QLineEdit.Password)
            groq_api_key_layout = QHBoxLayout()
            groq_api_key_layout.addWidget(groq_api_key_input)

            groq_toggle_btn = QPushButton("Show")
            groq_toggle_btn.clicked.connect(lambda: self.toggle_visibility(groq_api_key_input, groq_toggle_btn))
            groq_api_key_layout.addWidget(groq_toggle_btn)

            groq_layout.addRow("GROQ API Key:", groq_api_key_layout)
            groq_group.setLayout(groq_layout)
            layout.addWidget(groq_group)

            # Save Button
            save_button = QPushButton("Save Settings")
            save_button.setStyleSheet("background-color: green; color: white;")
            save_button.clicked.connect(
                lambda: self.save_settings(
                    chatgpt_url_input, chatgpt_api_key_input, ha_url_input, ha_api_key_input,
                    user_name_input, photo_path, audio_file_input, hf_token_input, groq_api_key_input
                )
            )
            layout.addWidget(save_button)

            # Add spacer at bottom
            layout.addStretch()

            page.setLayout(layout)
            return page

    def upload_photo(self, photo_label):
            file_path, _ = QFileDialog.getOpenFileName(None, "Select Photo", "", "Images (*.png *.jpg *.jpeg)")
            if file_path:
                self.save_to_env("USER_PHOTO", file_path)
                pixmap = QPixmap(file_path)
                photo_label.setPixmap(pixmap.scaled(100, 100))

    def upload_audio(self, audio_file_input):
            file_path, _ = QFileDialog.getOpenFileName(None, "Select Audio File", "", "Audio Files (*.mp3 *.wav)")
            if file_path:
                self.save_to_env("USER_AUDIO_FILE", file_path)
                audio_file_input.setText(file_path)

    def save_settings(self, chatgpt_url, chatgpt_api_key, ha_url, ha_api_key, user_name, photo_path, audio_file,
                          hf_token, groq_api_key):
            self.save_to_env("NEXUSAI_BASE_URL", chatgpt_url.text())
            self.save_to_env("NEXUSAI_API_KEY", chatgpt_api_key.text())
            self.save_to_env("HOME_ASSISTANT_URL", ha_url.text())
            self.save_to_env("HOME_ASSISTANT_API_KEY", ha_api_key.text())
            self.save_to_env("USER_NAME", user_name.text())
            self.save_to_env("USER_PHOTO", photo_path)
            self.save_to_env("USER_AUDIO_FILE", audio_file.text())
            self.save_to_env("HF_TOKEN", hf_token.text())
            self.save_to_env("GROQ_API_KEY", groq_api_key.text())
            QMessageBox.information(None, "Settings Saved", "Settings have been saved successfully.")

    def upload_json_file(self, button):
        from env import CACHE_DIR
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select JSON File", "", "JSON Files (*.json)")
        with open(file_path,"r") as f:
            data=f.read()
        if not os.path.exists(os.path.join(str(CACHE_DIR),"agent_files")):
            os.makedirs(os.path.join(str(CACHE_DIR),"agent_files"),exist_ok=True)
        with open(f"{os.path.join(str(CACHE_DIR),"agent_files",os.path.basename(file_path))}","w") as f:
            f.write(data)
        if file_path:
            button.setText(os.path.basename(file_path))

    def update_selected_api(self, text):
        self.selected_api = text
        self.update_home_page_configurations()

    def update_selected_audio_engine(self, text):
        self.selected_audio_engine = text
        self.update_home_page_configurations()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
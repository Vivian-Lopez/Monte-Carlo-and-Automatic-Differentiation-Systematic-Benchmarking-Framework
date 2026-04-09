import React, { useState, useCallback } from "react";
import { Box, Toolbar, CssBaseline } from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import Sidebar, { DRAWER_WIDTH } from "./components/Sidebar";
import TopBar from "./components/TopBar";
import SimulatePage from "./pages/SimulatePage";
import HistoryPage from "./pages/HistoryPage";
import SummaryPage from "./pages/SummaryPage";
import AdAnalysisPage from "./pages/AdAnalysisPage";
import GpuPage from "./pages/GpuPage";
import ScalingPage from "./pages/ScalingPage";
import CloudPage from "./pages/CloudPage";

const theme = createTheme({
    palette: {
        mode: "light",
        primary: { main: "#1565c0" },
        background: { default: "#f5f6fa", paper: "#ffffff" },
    },
    typography: {
        fontFamily: '"Inter", "Roboto", sans-serif',
    },
    shape: { borderRadius: 8 },
    components: {
        MuiCard: {
            defaultProps: { elevation: 0 },
        },
    },
});

const PAGE_TITLES: Record<string, string> = {
    simulate: "Run Simulation",
    history: "Run History",
    summary: "Dashboard",
    ad: "AD Analysis",
    gpu: "GPU Implementation",
    scaling: "Parallelism & Scaling",
    cloud: "Cloud Profiling",
};

export default function App() {
    const [page, setPage] = useState("simulate");
    const [backendStatus, setBackendStatus] = useState<"unknown" | "online" | "offline">("unknown");

    const handleBackendStatus = useCallback(
        (s: "online" | "offline") => setBackendStatus(s),
        []
    );

    function renderPage() {
        switch (page) {
            case "simulate":
                return <SimulatePage onBackendStatus={handleBackendStatus} />;
            case "history":
                return <HistoryPage />;
            case "summary":
                return <SummaryPage />;
            case "ad":
                return <AdAnalysisPage />;
            case "gpu":
                return <GpuPage />;
            case "scaling":
                return <ScalingPage />;
            case "cloud":
                return <CloudPage />;
            default:
                return null;
        }
    }

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Box sx={{ display: "flex" }}>
                <Sidebar currentPage={page} onNavigate={setPage} />
                <TopBar title={PAGE_TITLES[page] ?? ""} backendStatus={backendStatus} />
                <Box
                    component="main"
                    sx={{
                        flexGrow: 1,
                        p: 3,
                        width: `calc(100% - ${DRAWER_WIDTH}px)`,
                        minHeight: "100vh",
                        bgcolor: "background.default",
                    }}
                >
                    <Toolbar />
                    {renderPage()}
                </Box>
            </Box>
        </ThemeProvider>
    );
}

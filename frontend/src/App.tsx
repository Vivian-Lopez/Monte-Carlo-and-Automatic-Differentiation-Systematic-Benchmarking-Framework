import React, { useState, useCallback } from "react";
import { Box, Toolbar, CssBaseline } from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import Sidebar, { DRAWER_WIDTH } from "./components/Sidebar";
import TopBar from "./components/TopBar";
import SimulatePage from "./pages/SimulatePage";
import ComparePage from "./pages/ComparePage";
import HistoryPage from "./pages/HistoryPage";
import SummaryPage from "./pages/SummaryPage";
import AnalysisPage from "./pages/AnalysisPage";

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
    compare: "Engine Comparison",
    history: "Run History",
    summary: "Dashboard",
    analysis: "Analysis",
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
            case "compare":
                return <ComparePage />;
            case "history":
                return <HistoryPage />;
            case "summary":
                return <SummaryPage />;
            case "analysis":
                return <AnalysisPage />;
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

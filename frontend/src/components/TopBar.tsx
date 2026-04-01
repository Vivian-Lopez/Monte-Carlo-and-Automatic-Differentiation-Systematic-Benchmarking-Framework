import React from "react";
import { AppBar, Toolbar, Typography, Chip, Box } from "@mui/material";
import { DRAWER_WIDTH } from "./Sidebar";

interface Props {
    title: string;
    backendStatus: "unknown" | "online" | "offline";
}

export default function TopBar({ title, backendStatus }: Props) {
    const statusColor =
        backendStatus === "online"
            ? "success"
            : backendStatus === "offline"
                ? "error"
                : "default";

    const statusLabel =
        backendStatus === "online"
            ? "API Online"
            : backendStatus === "offline"
                ? "API Offline"
                : "Checking…";

    return (
        <AppBar
            position="fixed"
            elevation={0}
            sx={{
                width: `calc(100% - ${DRAWER_WIDTH}px)`,
                ml: `${DRAWER_WIDTH}px`,
                bgcolor: "background.paper",
                borderBottom: "1px solid",
                borderColor: "divider",
                color: "text.primary",
            }}
        >
            <Toolbar>
                <Typography variant="h6" fontWeight={600} sx={{ flexGrow: 1 }}>
                    {title}
                </Typography>
                <Box>
                    <Chip label={statusLabel} color={statusColor} size="small" />
                </Box>
            </Toolbar>
        </AppBar>
    );
}

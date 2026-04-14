import React from "react";
import {
    Drawer,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Toolbar,
    Typography,
    Divider,
    Box,
} from "@mui/material";
import BarChartIcon from "@mui/icons-material/BarChart";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import HistoryIcon from "@mui/icons-material/History";
import CompareArrowsIcon from "@mui/icons-material/CompareArrows";
import AnalyticsIcon from "@mui/icons-material/Analytics";

export const DRAWER_WIDTH = 240;

interface Props {
    currentPage: string;
    onNavigate: (page: string) => void;
}

const NAV_ITEMS = [
    { id: "simulate", label: "Run Simulation", icon: <PlayArrowIcon /> },
    { id: "compare", label: "Compare", icon: <CompareArrowsIcon /> },
    { id: "history", label: "Run History", icon: <HistoryIcon /> },
    { id: "summary", label: "Dashboard", icon: <BarChartIcon /> },
    { id: "analysis", label: "Analysis", icon: <AnalyticsIcon /> },
];

export default function Sidebar({ currentPage, onNavigate }: Props) {
    return (
        <Drawer
            variant="permanent"
            sx={{
                width: DRAWER_WIDTH,
                flexShrink: 0,
                "& .MuiDrawer-paper": {
                    width: DRAWER_WIDTH,
                    boxSizing: "border-box",
                    bgcolor: "background.paper",
                    borderRight: "1px solid",
                    borderColor: "divider",
                },
            }}
        >
            <Toolbar disableGutters sx={{ px: 2, py: 1.5 }}>
                <Box display="flex" alignItems="center" gap={1}>
                    <BarChartIcon color="primary" />
                    <Typography variant="subtitle1" fontWeight={700} noWrap>
                        MC Benchmark
                    </Typography>
                </Box>
            </Toolbar>
            <Divider />
            <List dense>
                {NAV_ITEMS.map((item) => (
                    <ListItem key={item.id} disablePadding>
                        <ListItemButton
                            selected={currentPage === item.id}
                            onClick={() => onNavigate(item.id)}
                            sx={{
                                borderRadius: 1,
                                mx: 0.5,
                                my: 0.25,
                                "&.Mui-selected": {
                                    bgcolor: "primary.main",
                                    color: "primary.contrastText",
                                    "& .MuiListItemIcon-root": { color: "primary.contrastText" },
                                    "&:hover": { bgcolor: "primary.dark" },
                                },
                            }}
                        >
                            <ListItemIcon sx={{ minWidth: 36 }}>{item.icon}</ListItemIcon>
                            <ListItemText
                                primary={item.label}
                                primaryTypographyProps={{ fontSize: 14 }}
                            />
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
        </Drawer>
    );
}

import React from "react";
import {
    Chip,
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
import FunctionsIcon from "@mui/icons-material/Functions";
import MemoryIcon from "@mui/icons-material/Memory";
import ScaleIcon from "@mui/icons-material/Scale";
import CloudIcon from "@mui/icons-material/Cloud";

export const DRAWER_WIDTH = 240;

interface Props {
    currentPage: string;
    onNavigate: (page: string) => void;
}

const NAV_ITEMS = [
    { id: "simulate", label: "Run Simulation", icon: <PlayArrowIcon />, status: "active" },
    { id: "history", label: "Run History", icon: <HistoryIcon />, status: "active" },
    { id: "summary", label: "Dashboard", icon: <BarChartIcon />, status: "active" },
    { id: "ad", label: "AD Analysis", icon: <FunctionsIcon />, status: "active" },
    { id: "gpu", label: "GPU", icon: <MemoryIcon />, status: "soon" },
    { id: "scaling", label: "Scaling", icon: <ScaleIcon />, status: "soon" },
    { id: "cloud", label: "Cloud", icon: <CloudIcon />, status: "soon" },
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
                            {item.status === "soon" && (
                                <Chip
                                    label="Soon"
                                    size="small"
                                    sx={{ height: 16, fontSize: 9, ml: 0.5 }}
                                />
                            )}
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
        </Drawer>
    );
}
